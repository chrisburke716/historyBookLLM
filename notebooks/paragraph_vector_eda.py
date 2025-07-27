import marimo

__generated_with = "0.14.13"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import weaviate
    from weaviate.classes.query import Sort

    # Use the fixed repository pattern instead of direct client
    from history_book.database.repositories import BookRepositoryManager
    from history_book.database.config import WeaviateConfig

    return BookRepositoryManager, WeaviateConfig, np


@app.cell
def _(BookRepositoryManager, WeaviateConfig):
    config = WeaviateConfig()
    repos = BookRepositoryManager(config)

    # Get all paragraphs sorted by indices
    paragraphs = repos.paragraphs.list_all()

    # Sort them manually since list_all doesn't support sorting yet
    paragraphs_sorted = sorted(
        paragraphs, 
        key=lambda p: (p.book_index, p.chapter_index, p.paragraph_index)
    )

    print(f"Total paragraphs loaded: {len(paragraphs_sorted)}")
    print(f"First paragraph: book={paragraphs_sorted[0].book_index}, chapter={paragraphs_sorted[0].chapter_index}, para={paragraphs_sorted[0].paragraph_index}")
    print(f"Last paragraph: book={paragraphs_sorted[-1].book_index}, chapter={paragraphs_sorted[-1].chapter_index}, para={paragraphs_sorted[-1].paragraph_index}")

    return (paragraphs_sorted,)


@app.cell
def _():
    # Clean up connections at the end of the notebook
    # repos.close_all()
    return


@app.cell
def _(paragraphs_sorted):
    type(paragraphs_sorted[0]) if paragraphs_sorted else "No paragraphs"
    return


@app.cell
def _(paragraphs_sorted):
    len(paragraphs_sorted)
    return


@app.cell
def _(paragraphs_sorted):
    paragraphs_sorted[0].embedding
    return


@app.cell
def _(paragraphs_sorted):
    # Extract vectors and match them to our sorted paragraphs by ID
    # vector_by_id = {}
    # for obj in result.objects:
    #     if obj.vector and 'text_vector' in obj.vector:
    #         vector_by_id[str(obj.uuid)] = np.array(obj.vector['text_vector'])

    # Create ordered vector list matching paragraphs_sorted
    vec_list = []
    missing_count = 0
    for paragraph in paragraphs_sorted:
        # if paragraph.id in vector_by_id:
        #     vec_list.append(vector_by_id[paragraph.id])
        # else:
        #     missing_count += 1
        if paragraph.embedding is not None:
            vec_list.append(paragraph.embedding)
        else:
            missing_count += 1

    # client.close()
    print(f"Extracted {len(vec_list)} vectors from {len(paragraphs_sorted)} paragraphs")
    if missing_count > 0:
        print(f"Warning: {missing_count} paragraphs missing vectors")
    return (vec_list,)


@app.cell
def _(np, vec_list):
    vec_mat = np.stack(vec_list)
    return (vec_mat,)


@app.cell
def _(np, vec_mat):
    # Normalize vectors first for proper cosine similarity
    vec_mat_norm = vec_mat / np.linalg.norm(vec_mat, axis=1, keepdims=True)
    cos_mat = np.dot(vec_mat_norm, vec_mat_norm.T)
    return (cos_mat,)


@app.cell
def _(cos_mat):
    cos_mat
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    return (plt,)


@app.cell
def _():
    # todo: add book, chapter delimeters to above
    return


@app.cell
def _(paragraphs_sorted):
    paragraph_books = [p.book_index for p in paragraphs_sorted]
    # count paragraphs per book
    from collections import Counter
    book_counts = Counter(paragraph_books)
    print("Paragraph counts per book:")
    for book, count in book_counts.items():
        print(f"Book {book}: {count} paragraphs")
    return Counter, book_counts


@app.cell
def _(book_counts, np):
    # get cumsum of book counts for plotting
    book_cumsum = np.cumsum([0] + [book_counts[k] for k in book_counts.keys()])
    print("Cumulative paragraph counts per book:")
    for _book, _cumsum in zip(book_counts.keys(), book_cumsum[1:]):
        print(f"Book {_book}: {_cumsum} cumulative paragraphs")
    return (book_cumsum,)


@app.cell
def _(Counter, paragraphs_sorted):
    # get chapter indices for each paragraph
    # need to note book too
    chapter_indices = [(p.book_index, p.chapter_index) for p in paragraphs_sorted]
    # count paragraphs per chapter
    chapter_counts = Counter(chapter_indices)
    print("Paragraph counts per chapter:")
    for (_book, _chapter), _count in chapter_counts.items():
        print(f"Book {_book}, Chapter {_chapter}: {_count} paragraphs")
    return (chapter_counts,)


@app.cell
def _(chapter_counts, np):
    # get cumsum of chapter counts for plotting
    chapter_cumsum = np.cumsum([0] + [chapter_counts[(k[0], k[1])] for k in chapter_counts.keys()])
    print("Cumulative paragraph counts per chapter:")
    for (_book, _chapter), _cumsum in zip(chapter_counts.keys(), chapter_cumsum[1:]):
        print(f"Book {_book}, Chapter {_chapter}: {_cumsum} cumulative paragraphs")
    return (chapter_cumsum,)


@app.cell
def _():
    from matplotlib import cm, colors
    return cm, colors


@app.cell
def _(book_cumsum, chapter_cumsum, cm, colors, cos_mat, plt):
    # increase figure size for better visibility
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    ax.matshow(cos_mat, cmap='viridis')
    # add horizontal and vertical lines to demarcate books
    for _cumsum in book_cumsum[1:]:
        ax.axhline(y=_cumsum - 0.5, color='red', linestyle='--')
        ax.axvline(x=_cumsum - 0.5, color='red', linestyle='--')
    # add horizontal and vertical lines to demarcate chapters
    # use thin black lines for chapter boundaries
    for _cumsum in chapter_cumsum[1:]:
        ax.axhline(y=_cumsum - 0.5, color='red', linestyle=':', linewidth=1)
        ax.axvline(x=_cumsum - 0.5, color='red', linestyle=':', linewidth=1)
    ax.set_title("Cosine Similarity Matrix of Paragraph Vectors")
    ax.set_xlabel("Paragraph Index")
    ax.set_ylabel("Paragraph Index")
    cmap = plt.get_cmap("viridis")
    norm = colors.Normalize(0, 5)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    fig.gca()
    return


@app.cell
def _(book_cumsum):
    book_cumsum
    return


@app.cell
def _(chapter_cumsum):
    # focus on book 2, chapter 4
    chapter_cumsum
    return


@app.cell
def _(cm, cos_mat, plt):
    cos_mat_filtered = cos_mat[613:698, 613:698]  # Book 2, Chapter 4
    # _fig = plt.figure(figsize=(7, 7))
    _fig = plt.figure()
    _ax = _fig.add_subplot(111)
    _ax.matshow(cos_mat_filtered, cmap='viridis')
    _ax.set_title("Cosine Similarity Matrix of Paragraph Vectors")
    _ax.set_xlabel("Paragraph Index")
    _ax.set_ylabel("Paragraph Index")
    _cmap = plt.get_cmap("viridis")
    # _norm = colors.Normalize(0, 1)
    _fig.colorbar(cm.ScalarMappable(norm=None, cmap=_cmap), ax=_ax)
    _fig.gca()
    return (cos_mat_filtered,)


@app.cell
def _(cos_mat_filtered, plt):
    # line plot of consecutive paragraph similarity
    plt.figure(figsize=(12, 6))
    for _i in range(1, 6, 2):
        plt.plot(cos_mat_filtered.diagonal(_i), marker='o', label=f"Diagonal {_i}")
    plt.title("Consecutive Paragraph Similarity in Book 2, Chapter 4")
    plt.xlabel("Consecutive Paragraph Index")
    plt.ylabel("Cosine Similarity")
    plt.grid()
    # plt.xticks(range(len(cos_mat_filtered)), range(613, 698), rotation=45)
    # plt.tight_layout()
    plt.legend()
    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
