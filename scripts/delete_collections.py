
from history_book.database import server
from history_book.database.config.database_config import WeaviateConfig

print("WARNING: This script will delete all collections in the Weaviate database.")
print("do you want to continue? (yes/no)")
user_input = input().strip().lower()
if user_input != "yes":
    print("Aborting...")
    exit(0)

config = WeaviateConfig.from_environment()

client = server.get_client(config)

for collection_name in client.collections.list_all().keys():
    print(f"Deleting existing collection: {collection_name}")
    client.collections.delete(collection_name)

print("All collections deleted successfully.")
print("You can now run the ingestion script to recreate them.")
print("Run `python scripts/run_ingestion.py` to start the ingestion process.")
server.close_client()
exit(0)
