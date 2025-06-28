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

    text: str
    embedding: Optional[List[float]] = (
        None  # Will be populated later during embedding generation
    )
    page: int
    paragraph_index: int
    book_index: int
    chapter_index: int
