# from weaviate.classes.config import DataType
from typing import Optional
from pydantic import BaseModel
from weaviate.collections.collection.sync import Collection

from weaviate.classes.config import Configure, Property, DataType
from pydantic.fields import FieldInfo

from weaviate import WeaviateClient


# first attempt - from copilot - trouble with model_json_schema
def create_collection_from_pydantic_old(
    client: WeaviateClient,
    model_class: BaseModel,
    # collection_name=None,
    references=None,
    vectorize_fields=None,
) -> Collection:
    """
    Create a Weaviate collection directly from a Pydantic model

    Args:
        client: Weaviate client
        model_class: Pydantic model class
        collection_name: Optional name for the collection (defaults to model class name)
        references: Optional list of references configurations
        vectorize_fields: Optional list of field names to vectorize

    Returns:
        Collection: The created Weaviate collection

    """
    # if collection_name is None:
    #     collection_name = model_class.__name__.replace("DBModel", "")
    collection_name = model_class.collection_name

    # Get model schema
    # TODO: fix this, can't get schema for model_class
    schema = model_class.model_json_schema()
    properties = []

    # Map Pydantic/Python types to Weaviate data types
    type_mapping = {
        "string": DataType.TEXT,
        "integer": DataType.INT,
        "number": DataType.NUMBER,
        "boolean": DataType.BOOL,
        "array": DataType.TEXT_ARRAY,
        "object": DataType.OBJECT,
    }

    # Check if collection exists and delete
    # TODO: if collection is going to be deleted, should make sure this is only called when necessary
    if collection_name in client.collections.list_all().keys():
        client.collections.delete(collection_name)

    # Extract properties from the model schema
    for field_name, field_info in schema.get("properties", {}).items():
        # Skip ID field as Weaviate handles this
        if field_name == "id":
            continue

        field_type = field_info.get("type")
        weaviate_type = type_mapping.get(field_type, DataType.TEXT)

        # Skip embedding field as Weaviate handles vectors
        if field_name == "embedding":
            continue

        property_config = {
            "name": field_name,
            "data_type": weaviate_type,
            "description": f"The {field_name} of the {collection_name.lower()}",
        }

        # Add vectorization config if requested
        if vectorize_fields and field_name in vectorize_fields:
            property_config["moduleConfig"] = {
                "text2vec-transformers": {"vectorize": True}
            }

        properties.append(property_config)

    # Create collection
    collection_config = {"name": collection_name, "properties": properties}

    # Add vectorizer if we have fields to vectorize
    if vectorize_fields:
        collection_config["vectorizer"] = "text2vec-transformers"

    # Add references if provided
    if references:
        collection_config["references"] = references

    # Create the collection
    collection = client.collections.create(**collection_config)

    return collection


def pydantic_field_to_weaviate_property(
    field_name: str, field_info: FieldInfo
) -> Optional[Property]:
    # Ignore internal fields like embeddings or anything you don't want in schema
    if field_name == "embedding":
        return None

    if field_name == "id":
        return None

    # Get the annotation type from the field_info
    annotation = getattr(field_info, "annotation", None)

    # Map Python types to Weaviate DataType
    if annotation == str:
        dtype = DataType.TEXT
    elif annotation == int:
        dtype = DataType.INT
    elif annotation == float:
        dtype = DataType.NUMBER
    elif annotation == bool:
        dtype = DataType.BOOL
    else:
        # Unknown or unsupported type
        return None

    return Property(name=field_name, data_type=dtype)


def create_collection_from_pydantic(
    client: WeaviateClient, model_class: type[BaseModel], class_name: str
) -> Collection:
    model_fields = model_class.model_fields
    properties = [
        prop
        for field_name, field_info in model_fields.items()
        if (prop := pydantic_field_to_weaviate_property(field_name, field_info))
        is not None
    ]

    # add vectorization config if needed, references
    vectorize_fields = getattr(model_class, "vectorize_fields", None)
    references = getattr(model_class, "references", None)
    vectorizer_config = []
    if vectorize_fields:
        for prop in properties:
            if prop.name in vectorize_fields:
                # prop.module_config = {"text2vec-openai": {"vectorize": True}}
                vectorizer_config.append(
                    Configure.NamedVectors.text2vec_openai(
                        name=f"{prop.name}_vector",
                        source_properties= [prop.name],
                        model="text-embedding-3-large",
                        dimensions=256,
                    )
                )

    collection_config = {"name": class_name, "properties": properties}
    if vectorizer_config:
        collection_config["vectorizer_config"] = vectorizer_config

    # print(f"Creating collection {class_name} with properties: {properties}")

    collection = client.collections.create(**collection_config)

    return collection
