"""Fetch paragraphs from Book 3, Chapter 4 (Rome chapter) for KG experimentation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from history_book.database.config.database_config import WeaviateConfig
from history_book.database.repositories.book_repository import BookRepositoryManager


def main():
    """Fetch and display paragraphs from Book 3, Chapter 4."""
    # Get database config
    config = WeaviateConfig.from_environment()

    # Initialize repository manager
    manager = BookRepositoryManager(config)

    # Fetch paragraphs from Book 3, Chapter 4
    # book_index=3 (1-indexed), chapter_index=4 (0-indexed where 0=Introduction)
    print("Fetching paragraphs from Book 3, Chapter 4 (Rome chapter)...")
    paragraphs = manager.paragraphs.find_by_chapter_index(book_index=3, chapter_index=4)

    print(f"\nFound {len(paragraphs)} paragraphs\n")
    print("=" * 80)

    # Display first 10 paragraphs with indices
    for i, para in enumerate(paragraphs[:10]):
        print(f"\n[Paragraph {i}] (ID: {para.id}, Page: {para.page})")
        print(f"Text: {para.text[:300]}...")  # First 300 chars
        print("-" * 80)

    if len(paragraphs) > 10:
        print(f"\n... and {len(paragraphs) - 10} more paragraphs")

    # Save all to JSON for notebook use
    import json
    output_file = Path(__file__).parent.parent.parent / "notebooks" / "book3_chapter4_paragraphs.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict
    paragraphs_data = [
        {
            "id": p.id,
            "text": p.text,
            "page": p.page,
            "paragraph_index": p.paragraph_index,
            "book_index": p.book_index,
            "chapter_index": p.chapter_index,
        }
        for p in paragraphs
    ]

    with open(output_file, "w") as f:
        json.dump(paragraphs_data, f, indent=2)

    print(f"\nSaved {len(paragraphs)} paragraphs to: {output_file}")


if __name__ == "__main__":
    main()
