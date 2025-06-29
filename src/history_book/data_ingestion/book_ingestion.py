import pymupdf
from history_book.data_models.book import ParagraphDBModel, ChapterDBModel, BookDBModel
from history_book.text_processing.text_processing import clean_text
from pathlib import Path

FINAL_PAGE = 1699  # last page of the book
PROJECT_ROOT = Path(__file__).parents[3]
BOOK_FILE = PROJECT_ROOT / "data" / "penguin_history_6.pdf"


def open_book(book_file: str) -> pymupdf.Document:
    """
    Opens a book file using pymupdf.

    Args:
        book_file (str): The path to the book file.

    Returns:
        pymupdf.Document: The opened book document.
    """
    doc = pymupdf.open(book_file)
    return doc


def get_chapter_starts(doc: pymupdf.Document) -> list[list[int]]:
    """
    Extracts the starting pages of each chapter from the history book.


    Args:
        doc (pymupdf.Document): The book document.

    Returns:
        book_start_pages: A list of lists, where each inner list contains the starting pages of chapters in a book.
            The final list contains the first page after the end of the last chapter.
    """
    book_start_pages = []
    chapter_start_pages = []
    for i, page in enumerate(doc):
        blocks = page.get_text("blocks")
        for block in blocks:
            if block[6] == 0:  # only text blocks
                text = block[4].strip()
                # check if text is a book header, e.g. 'Book One', 'Book Two', etc.
                if text.startswith("Book ") and len(text.split()) == 2:
                    # print(f"Found book header: {text}, on page {i+1}")
                    if len(chapter_start_pages) > 0:
                        # dont' add empty list at the start
                        book_start_pages.append(chapter_start_pages)
                    chapter_start_pages = [i]
                # for chapters: if first line is only a number, it's a chapter header
                elif (
                    text.isdigit() and len(text) < 3
                ):  # assuming chapter numbers are short
                    # print(f"Found chapter header: {text}, on page {i+1}")
                    chapter_start_pages.append(i)
    book_start_pages.append(chapter_start_pages)

    # add start of pages after last chapter
    book_start_pages.append([FINAL_PAGE])

    return book_start_pages


def combine_blocks(blocks: list, index1: int, index2: int):
    """
    Combines two text blocks into one by concatenating their text.
    Args:
        blocks (list): List of text blocks.
        index1 (int): Index of the first block to combine.
        index2 (int): Index of the second block to combine.
    """
    blocks[index1][4] += " " + blocks[index2][4]  # concatenate text
    del blocks[index2]  # remove the second block


def convert_blocks_to_lists(blocks: list) -> list[list]:
    """
    Converts a list of text blocks from tuples to lists.
    Args:
        blocks (list): List of text blocks as tuples.
    """
    return [list(block) for block in blocks]


def extract_chapter_titles(block: list) -> list[str]:
    """
    Extracts chapter titles from a block of text.
    Args:
        block (list): A block of text as a list.
    Returns:
        list: A list of chapter titles extracted from the block.
    """
    chapter_titles = []
    text = block[4].strip()
    # split by newlines and take the first part
    parts = text.split("\n")
    for part in parts:
        if part == "Introduction":
            # introductions are considered chapter 0
            chapter_titles.append("Introduction")
        # split by whitespace and take the second part (the chapter title)
        elif "\xa0\xa0" in part:
            title_part = part.split("\xa0\xa0")[1]
            chapter_titles.append(title_part.strip())
        else:
            # double digit chapter numbers might not have the '\xa0\xa0' separator
            # in that case remove the leading number and any whitespace
            title_part = part.lstrip("0123456789 ").strip()
            if title_part:
                chapter_titles.append(title_part)
    return chapter_titles


def organize_chapter_titles_by_book(chapter_titles: list[str]) -> list[list[str]]:
    """
    Groups chapter titles by book.
    Args:
        chapter_titles (list[str]): A flat list of chapter titles.
    Returns:
        list[list[str]]: A list of lists, where each inner list contains chapter titles for a book.
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


def get_chapter_titles_from_ToC(
    doc: pymupdf.Document,
) -> tuple[list[str], list[list[str]]]:
    """
    Extracts book and chapter titles from the Table of Contents (ToC) of the book.

    Returns:
        tuple: A tuple containing two lists:
            - book_titles: A list of book titles.
            - chapters_by_book: A list of lists, where each inner list contains chapter titles for a book.
    """

    toc_page_start = 4
    toc_page_end = 7

    # concat blocks into one list
    toc_blocks = []
    for i in range(toc_page_start - 1, toc_page_end):
        toc_blocks.extend(doc[i].get_text("blocks"))

    # need blocks mutable to combine them
    toc_blocks = convert_blocks_to_lists(toc_blocks)

    # combine ToC blocks that are split across pages
    # book 2 chapter block has "introduction" in it's own block
    combine_blocks(toc_blocks, 7, 8)
    # book 4 chapters
    combine_blocks(toc_blocks, 13, 14)
    # book 5 chapters
    combine_blocks(toc_blocks, 16, 17)
    # book 8 chapters
    combine_blocks(toc_blocks, 25, 26)

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
            chapter_titles.extend(extract_chapter_titles(block))

    # organize chapter titles by book
    chapters_by_book = organize_chapter_titles_by_book(chapter_titles)

    return book_titles, chapters_by_book


def get_chapter_text_blocks(
    doc: pymupdf.Document, start_page: int, end_page: int
) -> tuple[list[str], list[int]]:
    """
    Extracts text blocks from the specified range of pages in the book document.
    Args:
        doc (pymupdf.Document): The book document.
        start_page (int): The starting page number (1-indexed).
        end_page (int): The ending page number (1-indexed).
    Returns:
        tuple: A tuple containing two lists:
            - blocks: A list of cleaned text blocks.
            - block_pages: A list of page numbers corresponding to each text block.
    """
    blocks = []
    block_pages = []
    # page numbers are 1-indexed, so we need to adjust accordingly
    for n, page in enumerate(
        doc[start_page - 1 : end_page - 1]
    ):  # FIX: Adjust end_page to be exclusive
        page_blocks = page.get_text("blocks")
        cleaned_blocks = [
            clean_text(block[4]) for block in page_blocks if block[6] == 0
        ]  # only text blocks
        if n == 0:
            # if this is the first page, skip chapter header blocks
            # TODO: this works for chapter starts but not for book starts i.e. introductions
            cleaned_blocks = cleaned_blocks[2:]
        blocks.extend(cleaned_blocks)
        block_pages.extend([n + start_page] * len(cleaned_blocks))
    return blocks, block_pages


def concatenate_paragraphs(
    paragraphs: list[str], page_numbers: list[int]
) -> tuple[list[str], list[int]]:
    """
    Concatenates paragraphs that were split across pages, assuming that any paragraph
    that does not end with a period is continued on the next page.
    Returns a list of concatenated paragraphs and their corresponding page numbers.

    Args:
        paragraphs (list[str]): The list of paragraphs to concatenate.
        page_numbers (list[int]): The list of page numbers corresponding to each paragraph.

    Returns:
        tuple: A tuple containing two lists:
            - concatenated: A list of concatenated paragraphs.
            - concatenated_page_numbers: A list of page numbers corresponding to each concatenated paragraph.
    """
    concatenated = []
    concatenated_page_numbers = []
    previous_paragraph = ""
    previous_page = None

    for paragraph, page in zip(paragraphs, page_numbers):
        if previous_paragraph:
            # Check if the last character of the current paragraph is a period
            if previous_paragraph[-1] == ".":
                concatenated.append(previous_paragraph)
                concatenated_page_numbers.append(previous_page)
                previous_paragraph = paragraph
                previous_page = page
            else:
                previous_paragraph += " " + paragraph
        else:  # initialize
            previous_paragraph = paragraph
            previous_page = page

    if previous_paragraph:  # always append the last paragraph
        concatenated.append(previous_paragraph)
        concatenated_page_numbers.append(previous_page)

    return concatenated, concatenated_page_numbers


def create_paragraphs_db(
    text_blocks: list[str],
    page_numbers: list[int],
    chapter_index: int,
    book_index: int,
    start_index=0,
) -> list[ParagraphDBModel]:
    """
    Creates a list of ParagraphDBModel instances from the given text blocks and page numbers.
    Args:
        text_blocks (list[str]): A list of text blocks to be converted into paragraphs.
        page_numbers (list[int]): A list of page numbers corresponding to each text block.
        chapter_index (int): The index of the chapter to which these paragraphs belong.
        book_index (int): The index of the book to which these paragraphs belong.
        start_index (int): The starting index for paragraph numbering.
    Returns:
        list[ParagraphDBModel]: A list of ParagraphDBModel instances.
    """
    paragraphs = []

    for i, (block, page) in enumerate(
        zip(text_blocks, page_numbers), start=start_index
    ):
        paragraph = ParagraphDBModel(
            text=block,
            page=page,
            paragraph_index=i,
            chapter_index=chapter_index,
            book_index=book_index,
            embedding=None,  # Will be generated later
        )
        paragraphs.append(paragraph)

    return paragraphs


def process_chapter(
    doc: pymupdf.Document,
    chapter_title: str,
    chapter_index: int,
    start_page: int,
    end_page: int,
    book_index: int,
) -> tuple[ChapterDBModel, list[ParagraphDBModel]]:
    """Process a single chapter and extract its paragraphs

    Args:
        doc: PyMuPDF document
        chapter_title: Title of the chapter
        chapter_index: Index of chapter within its book
        start_page: 1-indexed start page
        end_page: 1-indexed end page (exclusive)
        book_index: Index of the book this chapter belongs to

    Returns:
        tuple: (chapter_model, paragraph_models)
    """
    blocks, block_pages = get_chapter_text_blocks(doc, start_page, end_page)

    # Join paragraphs that span across multiple blocks (i.e. split across pages)
    concatenated_blocks, concatenated_pages = concatenate_paragraphs(
        blocks, block_pages
    )

    # Create chapter DB model
    chapter = ChapterDBModel(
        title=chapter_title,
        start_page=start_page,
        end_page=end_page - 1,  # Store as inclusive end page
        chapter_index=chapter_index,
        book_index=book_index,
    )

    # Create paragraph DB models
    paragraphs = create_paragraphs_db(
        concatenated_blocks, concatenated_pages, chapter_index, book_index
    )

    return chapter, paragraphs


def process_book(
    doc: pymupdf.Document,
    book_index: int,
    book_title: str,
    book_start_pages: list[tuple[int]],
    chapter_titles: list[str],
) -> tuple[BookDBModel, list[ChapterDBModel], list[ParagraphDBModel]]:
    """Process a book and all its chapters

    Args:
        doc: PyMuPDF document
        book_index: Index of this book in the collection
        book_title: Title of the book
        book_start_pages: List of lists, where each inner list contains the starting pages of chapters in a book
        chapter_titles: List of chapter titles

    Returns:
        tuple: (book_model, chapter_models, paragraph_models)
    """
    # Calculate start and end pages for the book
    chapter_pages = book_start_pages[book_index]
    start_page = chapter_pages[0] + 1  # Convert 0-index to 1-index
    end_page = (
        book_start_pages[book_index + 1][0] + 1
    )  # Next book's start page (1-indexed)

    # Create book model
    book = BookDBModel(
        title=book_title,
        start_page=start_page,
        end_page=end_page - 1,  # Store as inclusive end page
        book_index=book_index,
    )

    # Process each chapter in this book
    all_chapters = []
    all_paragraphs = []

    for i, (chapter_title, chapter_page) in enumerate(
        zip(chapter_titles, chapter_pages)
    ):
        # Calculate chapter end page (start of next chapter or end of book)
        chapter_start = chapter_page + 1  # Convert to 1-indexed

        if i < len(chapter_pages) - 1:
            chapter_end = chapter_pages[i + 1] + 1  # Next chapter start (1-indexed)
        else:
            chapter_end = end_page  # End of book

        # Process the chapter
        chapter, paragraphs = process_chapter(
            doc, chapter_title, i, chapter_start, chapter_end, book_index
        )

        if chapter:
            all_chapters.append(chapter)
            all_paragraphs.extend(paragraphs)

    return book, all_chapters, all_paragraphs


def build_history_book_db() -> tuple[
    list[BookDBModel], list[ChapterDBModel], list[ParagraphDBModel]
]:
    """
    Build the history book database by processing the book file and extracting its structure.
    Returns:
        tuple: A tuple containing three lists:
            - all_books: List of BookDBModel instances.
            - all_chapters: List of ChapterDBModel instances.
            - all_paragraphs: List of ParagraphDBModel instances.
    """
    # Final collections for database
    all_books = []
    all_chapters = []
    all_paragraphs = []

    print("Opening book file...")
    doc = open_book(BOOK_FILE)

    print("Extracting book and chapter titles...")
    # Extract book and chapter titles from the Table of Contents
    book_titles, chapters_by_book = get_chapter_titles_from_ToC(doc)
    # Get the starting pages of each chapter
    book_start_pages = get_chapter_starts(doc)

    print("Process book data:")
    # Process each book (except last entry which is end marker)
    for book_index, (title, chapter_titles) in enumerate(
        zip(book_titles, chapters_by_book)
    ):
        # Process this book
        book, chapters, paragraphs = process_book(
            doc, book_index, title, book_start_pages, chapter_titles
        )

        # Add to our collections
        all_books.append(book)
        all_chapters.extend(chapters)
        all_paragraphs.extend(paragraphs)

        # Print progress
        print(f"Processed book {book_index + 1}/{len(book_titles)}: {title}")
        print(f"  Chapters: {len(chapters)}, Paragraphs: {len(paragraphs)}")

    return all_books, all_chapters, all_paragraphs
