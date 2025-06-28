import weaviate

# TODO: this should have a singleton pattern to avoid creating multiple clients

_client = None


def get_client() -> weaviate.WeaviateClient:
    """
    Create and return a Weaviate client instance.

    Returns:
        weaviate.WeaviateClient: An instance of the Weaviate client.
    """
    global _client
    if _client is None:
        # TODO: set up flag or environment variable to toggle test/prod
        _client = weaviate.connect_to_local()  # prod
        # _client = weaviate.connect_to_local(port=8081, grpc_port=50052)  # test server
    return _client
