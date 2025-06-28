from typing import List, Optional, ClassVar
from history_book.data_models.db_model import DBModel

# TODO: get rid of foreign keys, use book and chapter indices instead --- test in nb first


class BookDBModel(DBModel):
    collection_name: ClassVar[str] = "books"

    title: str
    start_page: int
    end_page: int
    book_index: int


class ChapterDBModel(DBModel):
    collection_name: ClassVar[str] = "chapters"

    title: str
    start_page: int
    end_page: int
    book_index: int
    chapter_index: int


class ParagraphDBModel(DBModel):
    collection_name: ClassVar[str] = "paragraphs"
    vectorize_fields: ClassVar[Optional[list[str]]] = ["text"]

    text: str
    embedding: Optional[List[float]] = (
        None  # Will be populated later during embedding generation
    )
    page: int
    paragraph_index: int
    book_index: int
    chapter_index: int

    def write_model_to_collection(self, reference_fields=None):
        """
        Override the parent method to handle vector embeddings
        """
        # Call the parent method to handle the basic insert
        result = super().write_model_to_collection(reference_fields)

        # After writing to database, extract the embedding that was created
        # and set it on the model instance
        if self.vectorize_fields:
            db_entry = self.collection.query.fetch_object_by_id(
                self.id, include_vector=True
            )
            # assuming only one vector field exists
            self.embedding = list(db_entry.vector.values())[0]

        return result
