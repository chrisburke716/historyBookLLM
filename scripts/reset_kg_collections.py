#!/usr/bin/env python3
"""Delete and recreate KG collections (KGEntities, KGRelationships, KGGraphs, KGMergeDecisions).

Use after schema changes to KG models. Does NOT touch Books/Chapters/Paragraphs.
"""

from history_book.database import server
from history_book.database.config.database_config import WeaviateConfig

# WeaviateRepository applies .capitalize() — "KGEntities" -> "Kgentities"
KG_COLLECTIONS = ["Kgentities", "Kgrelationships", "Kggraphs", "Kgmergedecisions"]

config = WeaviateConfig.from_environment()
client = server.get_client(config)

existing = set(client.collections.list_all())
for name in KG_COLLECTIONS:
    if name in existing:
        client.collections.delete(name)
        print(f"Deleted: {name}")
    else:
        print(f"Not found (skipping): {name}")

print("\nKG collections cleared. They'll be auto-recreated on next extraction run.")
server.close_client()
