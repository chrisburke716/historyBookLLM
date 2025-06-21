from typing import Optional, ClassVar
from pydantic import BaseModel, Field
from history_book.database.collections import create_collection_from_pydantic
from history_book.database.server import get_client
import uuid
from weaviate import WeaviateClient, Collection

class DBModel(BaseModel):
    """
    Base class for all database models.
    Provides common functionality for all models.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client: WeaviateClient = None
    collection: Optional[Collection] = None
    collection_name: ClassVar[str] = None

    # Pydantic having issues with Weaviate types, so we allow arbitrary types
    # --- not ideal, probably want to move client outside the model
    # TODO: check if this is still needed
    model_config = {
        "arbitrary_types_allowed": True  
    }

    def __init__(self, **data):
        super().__init__(**data)
        if self.client is None:
            self.client = get_client()
        self.set_collection()

    def create_collection(self) -> Collection:
        """
        Create a Weaviate collection for the model.
        """
        # self.collection = create_collection_from_pydantic(self.client, self)
        collection = create_collection_from_pydantic(self.client, self.__class__, self.collection_name)
        return collection

    def set_collection(self):
        """
        Set the Weaviate collection for the model.
        """
        if not self.collection_name:
            raise ValueError(f"Collection name not set for {self.__class__.__name__}")
            
        name_cap = self.collection_name.capitalize()
        # Check if the collection already exists
        if name_cap in self.client.collections.list_all().keys():
            self.collection = self.client.collections.get(name_cap)
        else:
            self.collection = self.create_collection()
            
        # Ensure we have a valid collection
        if self.collection is None:
            raise ValueError(f"Failed to create or retrieve collection {name_cap}")
        
        return self.collection

    def write_model_to_collection(self, reference_fields=None):
        """
        Write a Pydantic model to a Weaviate collection.
        
        Args:
            reference_fields: Optional dict mapping field names to reference configs
                            e.g. {'book_id': {'link_name': 'belongsToBook'}}
        
        Returns:
            The result from the collection insert operation
        """
        # Ensure we have a valid collection
        if self.collection is None:
            self.set_collection()
            
        if self.collection is None:
            raise ValueError(f"Failed to initialize collection for {self.__class__.__name__}")
        
        # Get model data as dict
        model_data = self.model_dump()
        
        # Extract UUID
        uuid = model_data.pop('id')
        
        # Filter out fields that shouldn't be written to the database
        filtered_data = {}
        model_fields = self.__class__.model_fields
        
        for field_name, value in model_data.items():
            # Skip internal fields that shouldn't be in the database
            if field_name in ['client', 'collection']:
                continue
                
            # Skip None values
            if value is None:
                continue
                
            # Only include fields that would be valid properties in Weaviate
            field_info = model_fields.get(field_name)
            if field_info:
                annotation = getattr(field_info, "annotation", None)
                # Only include primitive types that can be stored in Weaviate
                if annotation in [str, int, float, bool] or (
                    hasattr(annotation, "__origin__") and 
                    annotation.__origin__ is list and 
                    annotation.__args__[0] in [str, int, float, bool]
                ):
                    filtered_data[field_name] = value
        
        # Handle references separately if provided
        references = {}
        if reference_fields:
            for field_name, ref_config in reference_fields.items():
                if field_name in filtered_data:
                    link_name = ref_config.get('link_name')
                    references[link_name] = [filtered_data.pop(field_name)]
        
        # Insert into collection
        result = self.collection.data.insert(
            properties=filtered_data,
            uuid=uuid,
            references=references if references else None
        )
        
        return result
