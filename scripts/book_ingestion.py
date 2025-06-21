# script to automate the ingestion of the history book into the database
# import os
# import json
# from history_book.data_models.book import BookDBModel, ChapterDBModel, ParagraphDBModel
from history_book.data_ingestion.book_ingestion import build_history_book_db

if __name__ == "__main__":

    # Get book data (books, paragraphs, chapters)
    all_books, all_chapters, all_paragraphs = build_history_book_db()
    # TODO: fix - first page of every chapter is read twice: into previous chapter and into current chapter

    print("Ingesting history book data into the database...")
    # Write all models to their respective collections
    print("\tWriting books to collection...")
    for book in all_books:
        book.write_model_to_collection()
    print("\tWriting chapters to collection...")
    for chapter in all_chapters:
        chapter.write_model_to_collection()
    print("\tWriting paragraphs to collection...")
    for paragraph in all_paragraphs:
        paragraph.write_model_to_collection()

    print("History book data has been successfully ingested into the database.")


