"""Service for ingesting book data from PDF files into the vector database."""

from typing import List, Tuple, Optional
import logging
from pathlib import Path
import pymupdf

from ..data_models.entities import Book, Chapter, Paragraph
from ..database.repositories import BookRepositoryManager
from ..database.config import WeaviateConfig
from ..text_processing.text_processing import clean_text

logger = logging.getLogger(__name__)


class IngestionService:
    """Service for ingesting book data from PDF files into the vector database."""

    def __init__(self, config: Optional[WeaviateConfig] = None):
        """
        Initialize the ingestion service.

        Args:
            config: Optional WeaviateConfig. If not provided, will use environment-based config.
        """
        if config is None:
            config = WeaviateConfig.from_environment()
        self.config = config
        self._repo_manager: Optional[BookRepositoryManager] = None

    @property
    def repositories(self) -> BookRepositoryManager:
        """Get or create the repository manager."""
        if self._repo_manager is None:
            self._repo_manager = BookRepositoryManager(self.config)
        return self._repo_manager

    def clear_all_data(self) -> dict:
        """
        Clear all existing data from the database.

        This method removes all books, chapters, and paragraphs from the database.
        Use with caution as this operation cannot be undone.

        Returns:
            Dictionary with counts of deleted entities
        """
        logger.warning("⚠️  Clearing all data from database...")

        deleted_counts = {"books": 0, "chapters": 0, "paragraphs": 0}

        try:
            # Delete in order: paragraphs -> chapters -> books (to handle any dependencies)

            # Delete all paragraphs
            # TODO: this is pretty slow, should find a way to delete in bulk
            all_paragraphs = self.repositories.paragraphs.list_all()
            for paragraph in all_paragraphs:
                if paragraph.id:
                    self.repositories.paragraphs.delete(paragraph.id)
            deleted_counts["paragraphs"] = len(all_paragraphs)
            logger.info(f"Deleted {len(all_paragraphs)} paragraphs")

            # Delete all chapters
            all_chapters = self.repositories.chapters.list_all()
            for chapter in all_chapters:
                if chapter.id:
                    self.repositories.chapters.delete(chapter.id)
            deleted_counts["chapters"] = len(all_chapters)
            logger.info(f"Deleted {len(all_chapters)} chapters")

            # Delete all books
            all_books = self.repositories.books.list_all()
            for book in all_books:
                if book.id:
                    self.repositories.books.delete(book.id)
            deleted_counts["books"] = len(all_books)
            logger.info(f"Deleted {len(all_books)} books")

            total_deleted = sum(deleted_counts.values())
            logger.info(
                f"✅ Successfully cleared {total_deleted} entities from database"
            )

            return deleted_counts

        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            raise

    def check_existing_data(self) -> dict:
        """
        Check what data currently exists in the database.

        Returns:
            Dictionary with counts of existing entities
        """
        try:
            counts = {
                "books": self.repositories.books.count(),
                "chapters": self.repositories.chapters.count(),
                "paragraphs": self.repositories.paragraphs.count(),
            }
            return counts
        except Exception as e:
            logger.error(f"Failed to check existing data: {e}")
            return {"books": 0, "chapters": 0, "paragraphs": 0}

    def ingest_book_from_pdf(
        self, pdf_path: Path, final_page: int = 1699, clear_existing: bool = False
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Ingest a complete book from PDF into the database.

        Args:
            pdf_path: Path to the PDF file
            final_page: Last page number of the book
            clear_existing: Whether to clear existing data before ingestion

        Returns:
            Tuple of (book_ids, chapter_ids, paragraph_ids)
        """
        try:
            logger.info(f"Starting ingestion of book: {pdf_path}")

            # Check existing data
            existing_counts = self.check_existing_data()
            total_existing = sum(existing_counts.values())

            if total_existing > 0:
                logger.info(f"Found existing data: {existing_counts}")

                if clear_existing:
                    logger.info("Clearing existing data as requested...")
                    self.clear_all_data()
                else:
                    logger.warning(f"⚠️  Database contains {total_existing} entities!")
                    logger.warning("This may result in duplicate data.")
                    logger.warning(
                        "Consider using clear_existing=True or running the clear script first."
                    )

            # Step 1: Extract structure from PDF
            doc = self._open_book(pdf_path)
            book_titles, chapters_by_book = self._get_chapter_titles_from_toc(doc)
            book_start_pages = self._get_chapter_starts(doc, final_page)

            # Step 2: Process each book and create entities
            all_book_entities = []
            all_chapter_entities = []
            all_paragraph_entities = []

            # note: book_index is 1-indexed
            # We account for this in the entity creation
            for book_index, (book_title, chapter_titles) in enumerate(
                zip(book_titles, chapters_by_book)
            ):
                book_entity, chapter_entities, paragraph_entities = (
                    self._process_book_to_entities(
                        doc, book_index, book_title, book_start_pages, chapter_titles
                    )
                )

                all_book_entities.append(book_entity)
                all_chapter_entities.extend(chapter_entities)
                all_paragraph_entities.extend(paragraph_entities)

                logger.info(
                    f"Processed book {book_index + 1}/{len(book_titles)}: {book_title}"
                )
                logger.info(
                    f"  Chapters: {len(chapter_entities)}, Paragraphs: {len(paragraph_entities)}"
                )

            # Step 3: Save to database via repositories
            logger.info("Saving entities to database...")

            # Save books first
            book_ids = []
            for book in all_book_entities:
                book_id = self.repositories.books.create(book)
                book_ids.append(book_id)

            # Save chapters
            chapter_ids = []
            for chapter in all_chapter_entities:
                chapter_id = self.repositories.chapters.create(chapter)
                chapter_ids.append(chapter_id)

            # Save paragraphs in batches for efficiency
            # NOTE: using paragraph repo instead of service -- more symmetric with above but defeats the purpose of having service
            logger.info(f"Batch creating {len(all_paragraph_entities)} paragraphs...")
            paragraph_ids = self.repositories.paragraphs.batch_create_with_vectors(
                [
                    (p, None) for p in all_paragraph_entities
                ]  # Let Weaviate generate vectors
            )

            logger.info(
                f"Successfully ingested book with {len(book_ids)} books, {len(chapter_ids)} chapters, {len(paragraph_ids)} paragraphs"
            )

            return book_ids, chapter_ids, paragraph_ids

        except Exception as e:
            logger.error(f"Failed to ingest book: {e}")
            raise
        finally:
            if "doc" in locals():
                doc.close()

    def _open_book(self, book_file: Path) -> pymupdf.Document:
        """Open a book file using pymupdf."""
        return pymupdf.open(book_file)

    def _get_chapter_starts(
        self, doc: pymupdf.Document, final_page: int
    ) -> List[List[int]]:
        """
        Extract the starting pages of each chapter from the history book.

        Args:
            doc: The book document
            final_page: The last page of the book

        Returns:
            A list of lists, where each inner list contains the starting pages of chapters in a book.
        """
        book_start_pages = []
        chapter_start_pages = []

        for i, page in enumerate(doc):
            blocks = page.get_text("blocks")
            for block in blocks:
                if block[6] == 0:  # only text blocks
                    text = block[4].strip()

                    # Check if text is a book header, e.g. 'Book One', 'Book Two', etc.
                    if text.startswith("Book ") and len(text.split()) == 2:
                        if len(chapter_start_pages) > 0:
                            book_start_pages.append(chapter_start_pages)
                        chapter_start_pages = [i]

                    # For chapters: if first line is only a number, it's a chapter header
                    elif text.isdigit() and len(text) < 3:
                        chapter_start_pages.append(i)

        book_start_pages.append(chapter_start_pages)
        book_start_pages.append([final_page])  # Add end marker

        return book_start_pages

    def _get_chapter_titles_from_toc(
        self, doc: pymupdf.Document
    ) -> Tuple[List[str], List[List[str]]]:
        """
        Extract book and chapter titles from the Table of Contents.

        Returns:
            Tuple of (book_titles, chapters_by_book)
        """
        toc_page_start = 4
        toc_page_end = 7

        # Concat blocks into one list
        toc_blocks = []
        for i in range(toc_page_start - 1, toc_page_end):
            toc_blocks.extend(doc[i].get_text("blocks"))

        # Need blocks mutable to combine them
        toc_blocks = self._convert_blocks_to_lists(toc_blocks)

        # Combine ToC blocks that are split across pages
        # book 2 chapter block has "introduction" in it's own block
        self._combine_blocks(toc_blocks, 7, 8)
        # book 4 chapters
        self._combine_blocks(toc_blocks, 13, 14)
        # book 5 chapters
        self._combine_blocks(toc_blocks, 16, 17)
        # book 8 chapters
        self._combine_blocks(toc_blocks, 25, 26)

        book_titles = []
        chapter_titles = []

        for i, block in enumerate(toc_blocks):
            if i == 0:
                continue
            elif i == 1:
                continue
            elif i % 3 == 0:
                text = block[4].strip()
                book_titles.append(text)
            elif i % 3 == 1:
                chapter_titles.extend(self._extract_chapter_titles(block))

        # Organize chapter titles by book
        chapters_by_book = self._organize_chapter_titles_by_book(chapter_titles)

        return book_titles, chapters_by_book

    def _combine_blocks(self, blocks: List[List], index1: int, index2: int):
        """
        Combine two text blocks into one by concatenating their text.

        Args:
            blocks: List of text blocks
            index1: Index of the first block to combine
            index2: Index of the second block to combine
        """
        blocks[index1][4] += " " + blocks[index2][4]  # concatenate text
        del blocks[index2]  # remove the second block

    def _convert_blocks_to_lists(self, blocks: List) -> List[List]:
        """
        Convert a list of text blocks from tuples to lists.

        Args:
            blocks: List of text blocks as tuples

        Returns:
            List of text blocks as lists
        """
        return [list(block) for block in blocks]

    def _extract_chapter_titles(self, block: List) -> List[str]:
        """
        Extract chapter titles from a block of text.

        Args:
            block: A block of text as a list

        Returns:
            List of chapter titles extracted from the block
        """
        chapter_titles = []
        text = block[4].strip()
        # Split by newlines and take the first part
        parts = text.split("\n")
        for part in parts:
            if part == "Introduction":
                # Introductions are considered chapter 0
                chapter_titles.append("Introduction")
            # Split by whitespace and take the second part (the chapter title)
            elif "\xa0\xa0" in part:
                title_part = part.split("\xa0\xa0")[1]
                chapter_titles.append(title_part.strip())
            else:
                # Double digit chapter numbers might not have the '\xa0\xa0' separator
                # In that case remove the leading number and any whitespace
                title_part = part.lstrip("0123456789 ").strip()
                if title_part:
                    chapter_titles.append(title_part)
        return chapter_titles

    def _organize_chapter_titles_by_book(
        self, chapter_titles: List[str]
    ) -> List[List[str]]:
        """
        Group chapter titles by book.

        Args:
            chapter_titles: A flat list of chapter titles

        Returns:
            A list of lists, where each inner list contains chapter titles for a book
        """
        chapters_by_book = []

        # Find where each book's chapters start in the flat chapter_titles list
        book_chapter_indices = []
        current_idx = 0

        # Each book starts with "Introduction"
        for i, title in enumerate(chapter_titles):
            if title == "Introduction" and i > current_idx:
                book_chapter_indices.append(current_idx)
                current_idx = i

        # Add the last section
        book_chapter_indices.append(current_idx)
        # Add the end index
        book_chapter_indices.append(len(chapter_titles))

        # Create lists of chapter titles for each book
        for i in range(len(book_chapter_indices) - 1):
            start_idx = book_chapter_indices[i]
            end_idx = book_chapter_indices[i + 1]
            chapters_by_book.append(chapter_titles[start_idx:end_idx])

        return chapters_by_book

    def _process_book_to_entities(
        self,
        doc: pymupdf.Document,
        book_index: int,
        book_title: str,
        book_start_pages: List[List[int]],
        chapter_titles: List[str],
    ) -> Tuple[Book, List[Chapter], List[Paragraph]]:
        """
        Process a book and create pure entity objects (no database operations).

        Args:
            doc: PyMuPDF document
            book_index: Index of this book
            book_title: Title of the book
            book_start_pages: Chapter start pages
            chapter_titles: List of chapter titles

        Returns:
            Tuple of (book_entity, chapter_entities, paragraph_entities)
        """
        # Calculate book page range
        chapter_pages = book_start_pages[book_index]
        start_page = chapter_pages[0] + 1
        end_page = book_start_pages[book_index + 1][0] + 1

        # Create book entity (pure data, no database operations)
        book_entity = Book(
            title=book_title,
            start_page=start_page,
            end_page=end_page - 1,
            book_index=book_index + 1,  # 1-indexed
        )

        # Process chapters
        chapter_entities = []
        paragraph_entities = []

        for i, (chapter_title, chapter_page) in enumerate(
            zip(chapter_titles, chapter_pages)
        ):
            chapter_start = chapter_page + 1
            chapter_end = (
                chapter_pages[i + 1] + 1 if i < len(chapter_pages) - 1 else end_page
            )

            # Create chapter entity
            # Note: chapter_index is 0-indexed, book_index is 1-indexed
            # 'Chapter 0' is the introduction for each book
            chapter_entity = Chapter(
                title=chapter_title,
                start_page=chapter_start,
                end_page=chapter_end - 1,
                chapter_index=i,
                book_index=book_index + 1,  # 1-indexed
            )
            chapter_entities.append(chapter_entity)

            # Extract and create paragraph entities for this chapter
            chapter_paragraphs = self._extract_chapter_paragraphs(
                doc, chapter_start, chapter_end, i, book_index
            )
            paragraph_entities.extend(chapter_paragraphs)

        return book_entity, chapter_entities, paragraph_entities

    def _extract_chapter_paragraphs(
        self,
        doc: pymupdf.Document,
        start_page: int,
        end_page: int,
        chapter_index: int,
        book_index: int,
    ) -> List[Paragraph]:
        """
        Extract paragraph entities from a chapter.

        Args:
            doc: PyMuPDF document
            start_page: Chapter start page (1-indexed)
            end_page: Chapter end page (1-indexed, exclusive)
            chapter_index: Index of the chapter
            book_index: Index of the book

        Returns:
            List of paragraph entities
        """
        # Get text blocks from the chapter pages
        blocks, block_pages = self._get_chapter_text_blocks(doc, start_page, end_page)

        # Concatenate paragraphs that span multiple blocks
        concatenated_blocks, concatenated_pages = self._concatenate_paragraphs(
            blocks, block_pages
        )

        # Create paragraph entities
        # Note: only books are 1-indexed, chapters and paragraphs are 0-indexed
        # 'Chapter 0' for each book is the introduction
        paragraph_entities = []
        for i, (text_block, page) in enumerate(
            zip(concatenated_blocks, concatenated_pages)
        ):
            paragraph = Paragraph(
                text=text_block,
                page=page,
                paragraph_index=i,
                chapter_index=chapter_index,
                book_index=book_index + 1,  # 1-indexed
                embedding=None,  # Will be generated by Weaviate
            )
            paragraph_entities.append(paragraph)

        return paragraph_entities

    def _get_chapter_text_blocks(
        self, doc: pymupdf.Document, start_page: int, end_page: int
    ) -> Tuple[List[str], List[int]]:
        """Extract text blocks from chapter pages."""
        blocks = []
        block_pages = []

        for page_num in range(start_page - 1, end_page - 1):  # Convert to 0-indexed
            if page_num < len(doc):
                page = doc[page_num]
                page_blocks = page.get_text("blocks")

                for block in page_blocks:
                    if block[6] == 0:  # Text blocks only
                        text = clean_text(block[4].strip())
                        if text and len(text) > 10:  # Filter out very short blocks
                            blocks.append(text)
                            block_pages.append(
                                page_num + 1
                            )  # Convert back to 1-indexed

        return blocks, block_pages

    def _concatenate_paragraphs(
        self, blocks: List[str], block_pages: List[int]
    ) -> Tuple[List[str], List[int]]:
        """
        Concatenate paragraphs that are split across multiple blocks/pages.

        This is a simplified version - the original has more complex logic.
        """
        if not blocks:
            return [], []

        concatenated = []
        concatenated_pages = []

        current_paragraph = blocks[0]
        current_page = block_pages[0]

        for i in range(1, len(blocks)):
            block = blocks[i]
            page = block_pages[i]

            # Simple heuristic: if block starts with lowercase, it's likely a continuation
            if block and block[0].islower() and page == current_page + 1:
                current_paragraph += " " + block
            else:
                # Save current paragraph and start new one
                concatenated.append(current_paragraph)
                concatenated_pages.append(current_page)
                current_paragraph = block
                current_page = page

        # Don't forget the last paragraph
        if current_paragraph:
            concatenated.append(current_paragraph)
            concatenated_pages.append(current_page)

        return concatenated, concatenated_pages

    def close(self):
        """Close repository connections."""
        if self._repo_manager is not None:
            self._repo_manager.close_all()
            self._repo_manager = None
