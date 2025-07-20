import marimo

__generated_with = "0.14.12"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    from history_book.database.config import WeaviateConfig
    from history_book.database.repositories import BookRepositoryManager
    return BookRepositoryManager, WeaviateConfig


@app.cell
def _(WeaviateConfig):
    config = WeaviateConfig.from_environment()
    return (config,)


@app.cell
def _(BookRepositoryManager, config):
    repo_manager = BookRepositoryManager(config)
    return (repo_manager,)


@app.cell
def _(repo_manager):
    repo_manager.__dict__
    return


@app.cell
async def _(repo_manager):
    all_books = await repo_manager.books.list_all()
    all_books
    return (all_books,)


@app.cell
def _(all_books):
    all_books[0].id
    return


@app.cell
def _(all_books, repo_manager):
    repo_manager.books.get_by_id_sync(all_books[0].id).__dict__
    return


@app.cell
def _(repo_manager):
    # get chapter in book 1
    repo_manager.chapters.find_by_book_index_sync(1)
    return


@app.cell
def _(repo_manager):
    # check
    repo_manager.chapters.find_by_criteria_sync({'book_index':1})
    return


@app.cell
def _(repo_manager):
    # get paragraphs
    para_b2_c3 = repo_manager.paragraphs.find_by_chapter_index_sync(book_index=2, chapter_index=3)
    return (para_b2_c3,)


@app.cell
def _(para_b2_c3):
    para_b2_c3[0].__dict__
    return


@app.cell
def _(repo_manager):
    para_cleo = repo_manager.paragraphs.search_similar_paragraphs_sync(query_text="cleopatra")
    return (para_cleo,)


@app.cell
def _(para_cleo):
    para_cleo
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
