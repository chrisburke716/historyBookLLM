"""Knowledge Graph extraction pipeline CLI.

Thin wrapper around KGIngestionService. Uses Weaviate as sole storage layer.

Usage:
    # Extract single chapter
    poetry run python scripts/run_kg_extraction.py chapter --book 3 --chapter 2

    # Build book graph (auto-extracts missing chapters)
    poetry run python scripts/run_kg_extraction.py book --book 3 [--chapters 0 1 2]

    # Build volume graph from book graphs
    poetry run python scripts/run_kg_extraction.py volume --books 2 3 4 5 [--name full_volume]

    # Custom cross-book merge
    poetry run python scripts/run_kg_extraction.py custom --spec '{"2":[5,6],"3":[6,7]}' --name asia

    # List all graphs in DB
    poetry run python scripts/run_kg_extraction.py list
"""

import argparse
import json
import logging
import os
import time
import warnings

from history_book.services.kg_ingestion_service import KGIngestionService

os.environ["LANGCHAIN_TRACING_V2"] = "false"
warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger(__name__)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add flags common to all extraction/merge subcommands."""
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate results even if already in DB",
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
        "--similarity-threshold",
        type=float,
        default=None,
        help="Cosine similarity threshold for merge candidates (default: 0.65)",
    )


def cmd_chapter(args, service):
    """Extract a single chapter."""
    service.extract_chapter(
        args.book,
        args.chapter,
        force=args.force,
        profile=args.profile,
        max_concurrency=args.max_concurrency,
    )


def cmd_book(args, service):
    """Build a book graph (auto-extracts missing chapters)."""
    service.merge_book(
        args.book,
        chapters=args.chapters,
        force=args.force,
        profile=args.profile,
        max_concurrency=args.max_concurrency,
    )


def cmd_volume(args, service):
    """Build a volume graph from book graphs."""
    service.merge_volume(
        args.books,
        graph_name=args.name,
        force=args.force,
        profile=args.profile,
        max_concurrency=args.max_concurrency,
    )


def cmd_custom(args, service):
    """Custom cross-book merge from a JSON spec."""
    raw_spec = json.loads(args.spec)
    # Convert string keys to int
    chapters = {int(k): v for k, v in raw_spec.items()}
    service.merge_custom(
        chapters,
        graph_name=args.name,
        force=args.force,
        profile=args.profile,
        max_concurrency=args.max_concurrency,
    )


def cmd_list(args, service):
    """List all graphs in DB."""
    graphs = service.list_graphs()
    if not graphs:
        logger.info("No graphs found in database.")
        return
    logger.info(
        "%-25s %-10s %-8s %-8s %s", "Name", "Type", "Entities", "Rels", "Chapters"
    )
    logger.info("-" * 75)
    for g in sorted(graphs, key=lambda x: x.name):
        chapters_str = ", ".join(g.book_chapters[:10])
        if len(g.book_chapters) > 10:
            chapters_str += f" (+{len(g.book_chapters) - 10} more)"
        logger.info(
            "%-25s %-10s %-8d %-8d %s",
            g.name,
            g.graph_type,
            g.entity_count,
            g.relationship_count,
            chapters_str,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Graph extraction pipeline CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- chapter ---
    p_chapter = subparsers.add_parser(
        "chapter", help="Extract KG from a single chapter"
    )
    p_chapter.add_argument("--book", type=int, required=True, help="Book index")
    p_chapter.add_argument("--chapter", type=int, required=True, help="Chapter index")
    _add_common_args(p_chapter)

    # --- book ---
    p_book = subparsers.add_parser(
        "book", help="Build book graph (auto-extracts missing chapters)"
    )
    p_book.add_argument("--book", type=int, required=True, help="Book index")
    p_book.add_argument(
        "--chapters",
        type=int,
        nargs="+",
        default=None,
        help="Specific chapter indices (default: all chapters)",
    )
    _add_common_args(p_book)

    # --- volume ---
    p_volume = subparsers.add_parser(
        "volume", help="Build volume graph from book graphs"
    )
    p_volume.add_argument(
        "--books", type=int, nargs="+", required=True, help="Book indices to merge"
    )
    p_volume.add_argument(
        "--name", type=str, default=None, help="Custom graph name (default: auto)"
    )
    _add_common_args(p_volume)

    # --- custom ---
    p_custom = subparsers.add_parser("custom", help="Custom cross-book chapter merge")
    p_custom.add_argument(
        "--spec",
        type=str,
        required=True,
        help='JSON mapping: \'{"2":[5,6],"3":[6,7]}\'',
    )
    p_custom.add_argument(
        "--name", type=str, required=True, help="Graph name (required)"
    )
    _add_common_args(p_custom)

    # --- list ---
    subparsers.add_parser("list", help="List all graphs in DB")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Build pipeline config overrides
    pipeline_config = {}
    if hasattr(args, "similarity_threshold") and args.similarity_threshold is not None:
        pipeline_config["similarity_threshold"] = args.similarity_threshold

    service = KGIngestionService(pipeline_config=pipeline_config)
    t_start = time.perf_counter()
    try:
        commands = {
            "chapter": cmd_chapter,
            "book": cmd_book,
            "volume": cmd_volume,
            "custom": cmd_custom,
            "list": cmd_list,
        }
        commands[args.command](args, service)
    finally:
        elapsed = time.perf_counter() - t_start
        service.close()
        logger.info("Total wall time: %.1fs", elapsed)


if __name__ == "__main__":
    main()
