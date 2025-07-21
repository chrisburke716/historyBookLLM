# from weaviate.classes.config import DataType
from typing import Optional
from pydantic import BaseModel
from weaviate.collections.collection.sync import Collection

from weaviate.classes.config import Configure, Property, DataType
from pydantic.fields import FieldInfo

from weaviate import WeaviateClient


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
    if annotation is str:
        dtype = DataType.TEXT
    elif annotation is int:
        dtype = DataType.INT
    elif annotation is float:
        dtype = DataType.NUMBER
    elif annotation is bool:
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

    # Check if this model has vectorization fields
    vectorize_fields = getattr(model_class, "vectorize_fields", None)
    
    vectorizer_config = []
    if vectorize_fields:
        for prop in properties:
            if prop.name in vectorize_fields:
                vectorizer_config.append(
                    Configure.NamedVectors.text2vec_openai(
                        name=f"{prop.name}_vector",
                        source_properties=[prop.name],
                        model="text-embedding-3-large",
                        dimensions=256,
                    )
                )

    collection_config = {"name": class_name, "properties": properties}
    if vectorizer_config:
        collection_config["vectorizer_config"] = vectorizer_config

    collection = client.collections.create(**collection_config)

    return collection
