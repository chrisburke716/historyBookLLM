# script to automate the ingestion of the history book into the database
# import os
# import json
# from history_book.data_models.book import BookDBModel, ChapterDBModel, ParagraphDBModel
from history_book.data_ingestion.book_ingestion import build_history_book_db

"""
This script runs the full ingestion process for the history book.
It ingests the book text into book, chapter, and paragraph collections, 
and writes those to the database and generates embeddings for paragraphs.
"""

if __name__ == "__main__":
    # Get book data (books, paragraphs, chapters)
    build_history_book_db()
    print("History book data has been successfully ingested into the database.")
