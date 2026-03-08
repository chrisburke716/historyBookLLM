"""Knowledge Graph extraction pipeline CLI.

Thin wrapper around KGIngestionService. Processes paragraphs from the database,
extracts entities and relationships, normalizes via rule-based + embedding + LLM
merging, exports results to files, and stores in Weaviate.

Output structure:
    output/kg/chapters/book{X}_ch{Y}/    — centralized per-chapter cache
    output/kg/graphs/{name}/              — merged graph outputs with metadata

Single chapter (writes to centralized cache):
    poetry run python scripts/run_kg_extraction.py --book-index 3 --chapter-index 4

All chapters in a book (extract + merge):
    poetry run python scripts/run_kg_extraction.py --book-index 3

Specific chapters (extract + merge):
    poetry run python scripts/run_kg_extraction.py --book-index 3 --chapters 2 3

Incremental (add chapter to existing graph):
    poetry run python scripts/run_kg_extraction.py --book-index 3 --chapters 4 --base-graph output/kg/graphs/book3_ch2-3

Custom graph name:
    poetry run python scripts/run_kg_extraction.py --book-index 3 --chapters 2 3 4 --graph-name book3_greeks
"""

import argparse
import logging
import os
import time
import warnings

from history_book.database.config.database_config import WeaviateConfig
from history_book.database.repositories.book_repository import BookRepositoryManager
from history_book.services.kg_ingestion_service import KGIngestionService

os.environ["LANGCHAIN_TRACING_V2"] = "false"
warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run KG extraction pipeline on book chapter(s).",
        epilog="""Examples:
  # Single chapter extraction (writes to centralized cache)
  %(prog)s --book-index 3 --chapter-index 4

  # All chapters in a book (extract + merge)
  %(prog)s --book-index 3

  # Extract + merge specific chapters
  %(prog)s --book-index 3 --chapters 2 3

  # Add chapter 4 to an existing graph incrementally
  %(prog)s --book-index 3 --chapters 4 --base-graph output/kg/graphs/book3_ch2-3

  # Custom graph name
  %(prog)s --book-index 3 --chapters 2 3 4 --graph-name book3_greeks
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--book-index", type=int, required=True, help="Book index in the database"
    )
    parser.add_argument(
        "--chapter-index",
        type=int,
        default=None,
        help="Single chapter index (mutually exclusive with --chapters)",
    )
    parser.add_argument(
        "--chapters",
        type=int,
        nargs="+",
        default=None,
        help="Multiple chapter indices for cross-chapter merge",
    )
    parser.add_argument(
        "--base-graph",
        type=str,
        default=None,
        help="Path to a previous graph directory to extend incrementally",
    )
    parser.add_argument(
        "--graph-name",
        type=str,
        default=None,
        help="Custom name for the output graph directory (default: auto-generated)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Cosine similarity threshold for merge candidates (default: 0.65)",
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip PyVis HTML generation",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print timing summary for each pipeline stage",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help="Max parallel API calls for batch operations (default: 5)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate per-chapter results even if cached output exists",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence noisy HTTP request logs from OpenAI and Weaviate clients
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Validate mutual exclusivity
    if args.chapter_index is not None and args.chapters is not None:
        parser.error("--chapter-index and --chapters are mutually exclusive")
    if args.base_graph and args.chapter_index is not None:
        parser.error(
            "--base-graph can only be used with --chapters, not --chapter-index"
        )

    # If only --book-index given, discover all chapters for that book
    if args.chapter_index is None and args.chapters is None:
        mgr = BookRepositoryManager(WeaviateConfig.from_environment())
        chapters = mgr.chapters.find_by_book_index(args.book_index)
        mgr.close_all()
        if not chapters:
            parser.error(f"No chapters found for book index {args.book_index}")
        args.chapters = sorted(ch.chapter_index for ch in chapters)
        logger.info(
            "No chapters specified — running all %d chapters for book %d: %s",
            len(args.chapters),
            args.book_index,
            args.chapters,
        )

    # Build pipeline config overrides
    pipeline_config = {}
    if args.similarity_threshold is not None:
        pipeline_config["similarity_threshold"] = args.similarity_threshold

    service = KGIngestionService(pipeline_config=pipeline_config)
    t_start = time.perf_counter()
    try:
        if args.chapter_index is not None:
            service.extract_chapter(
                args.book_index,
                args.chapter_index,
                force=args.force,
                profile=args.profile,
                max_concurrency=args.max_concurrency,
                skip_visualization=args.skip_visualization,
            )
        else:
            service.merge_chapters(
                args.book_index,
                args.chapters,
                graph_name=args.graph_name,
                base_graph=args.base_graph,
                force=args.force,
                profile=args.profile,
                max_concurrency=args.max_concurrency,
                skip_visualization=args.skip_visualization,
            )
    finally:
        elapsed = time.perf_counter() - t_start
        service.close()
        logger.info("Total wall time: %.1fs", elapsed)


if __name__ == "__main__":
    main()
