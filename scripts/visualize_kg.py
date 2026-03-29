"""Standalone KG visualization script.

Loads a graph from Weaviate by name and generates a PyVis HTML visualization.

Usage:
    poetry run python scripts/visualize_kg.py book3_ch2
    poetry run python scripts/visualize_kg.py book3 --output book3_viz.html
"""

import argparse
import logging
import os
import warnings

from history_book.services.kg_ingestion_service import (
    KGIngestionService,
    build_knowledge_graph,
    visualize_with_pyvis,
)

os.environ["LANGCHAIN_TRACING_V2"] = "false"
warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a knowledge graph from Weaviate.",
    )
    parser.add_argument("graph_name", help="Name of the graph to visualize")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HTML file path (default: {graph_name}.html)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    service = KGIngestionService()
    try:
        result = service.load_graph(args.graph_name)
        if result is None:
            logger.error("Graph '%s' not found in database.", args.graph_name)
            # List available graphs
            graphs = service.list_graphs()
            if graphs:
                logger.info("Available graphs:")
                for g in sorted(graphs, key=lambda x: x.name):
                    logger.info(
                        "  %-25s (%s, %d entities)",
                        g.name,
                        g.graph_type,
                        g.entity_count,
                    )
            return

        entities, relationships = result
        output_file = args.output or f"{args.graph_name}.html"

        G = build_knowledge_graph(entities, relationships)
        logger.info(
            "Graph '%s': %d nodes, %d edges",
            args.graph_name,
            G.number_of_nodes(),
            G.number_of_edges(),
        )
        visualize_with_pyvis(G, entities, relationships, output_file)
    finally:
        service.close()


if __name__ == "__main__":
    main()
