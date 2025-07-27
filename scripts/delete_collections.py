# import weaviate

from history_book.database.config.database_config import WeaviateConfig

import weaviate

print("WARNING: This script will delete all collections in the Weaviate database.")
print("do you want to continue? (yes/no)")
user_input = input().strip().lower()
if user_input != "yes":
    print("Aborting...")
    exit(0)

config = WeaviateConfig.from_environment()

client = weaviate.connect_to_local(
        port=config.port,
        grpc_port=config.grpc_port
    )

for collection_name in client.collections.list_all().keys():
    print(f"Deleting existing collection: {collection_name}")
    client.collections.delete(collection_name)

print("All collections deleted successfully.")
print("You can now run the ingestion script to recreate them.")
print("Run `python scripts/run_ingestion.py` to start the ingestion process.")
client.close()
exit(0)