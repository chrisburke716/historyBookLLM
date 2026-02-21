"""Display 5 selected paragraphs for manual annotation."""

import json
from pathlib import Path

# IDs of selected paragraphs
SELECTED_IDS = [
    "0cd2eb74-f135-4c54-8d93-7ee75280146a",  # Rome's founding, Etruscans, geographic entities
    "1ef9dd6f-52bd-4f75-81f9-d3d6b8f53874",  # Caesar, political power, Senate
    "1fa5555b-bef9-4253-ad61-e2ffca2c5e22",  # Octavian/Augustus, titles, civil war
    "1cfb6550-4c6e-4c63-a181-7dc7a3a15e46",  # International relations: Augustus, Parthia, Trajan
    "22caab97-56ab-416f-9b1d-f785d2469418",  # Constitutional arrangements, Latin League, alliances
]


def main():
    """Display selected paragraphs with clear formatting."""
    # Load paragraphs
    json_file = (
        Path(__file__).parent.parent.parent
        / "notebooks"
        / "book3_chapter4_paragraphs.json"
    )
    with open(json_file) as f:
        all_paragraphs = json.load(f)

    # Filter selected
    selected = [p for p in all_paragraphs if p["id"] in SELECTED_IDS]

    # Display
    print("=" * 100)
    print("SELECTED PARAGRAPHS FOR MANUAL ANNOTATION")
    print("=" * 100)

    for i, para in enumerate(selected, 1):
        print(f"\n[PARAGRAPH {i}]")
        print(f"ID: {para['id']}")
        print(f"Page: {para['page']}")
        print("-" * 100)
        print(para["text"])
        print("-" * 100)
        print("\nENTITIES TO EXTRACT:")
        print("  - Persons (rulers, leaders, etc.)")
        print("  - Places (cities, regions, geographic features)")
        print("  - Political Entities (empires, republics, organizations)")
        print("  - Events (battles, treaties, foundings)")
        print("  - Time periods (dates, eras)")
        print("\nRELATIONSHIPS TO EXTRACT:")
        print("  - Who ruled what?")
        print("  - Who conquered what?")
        print("  - Who allied with whom?")
        print("  - What happened when?")
        print("  - Other notable relationships")
        print("=" * 100)

    # Save selected to separate JSON
    output_file = (
        Path(__file__).parent.parent.parent / "notebooks" / "selected_5_paragraphs.json"
    )
    with open(output_file, "w") as f:
        json.dump(selected, f, indent=2)

    print(f"\nSaved selected paragraphs to: {output_file}")


if __name__ == "__main__":
    main()
