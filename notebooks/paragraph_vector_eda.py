import marimo

__generated_with = "0.14.13"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    from collections import Counter
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    # Use the fixed repository pattern instead of direct client
    from history_book.database.repositories import BookRepositoryManager
    from history_book.database.config import WeaviateConfig

    return BookRepositoryManager, Counter, WeaviateConfig, cm, colors, np, plt


@app.cell
def _(mo):
    mo.md(r"""### Load paragraphs""")
    return


@app.cell
def _(BookRepositoryManager, WeaviateConfig):
    config = WeaviateConfig()
    repos = BookRepositoryManager(config)

    # Get all paragraphs sorted by indices
    paragraphs = repos.paragraphs.list_all()

    # Sort them manually since list_all doesn't support sorting yet
    paragraphs_sorted = sorted(
        paragraphs, key=lambda p: (p.book_index, p.chapter_index, p.paragraph_index)
    )

    print(f"Total paragraphs loaded: {len(paragraphs_sorted)}")
    print(
        f"First paragraph: book={paragraphs_sorted[0].book_index}, chapter={paragraphs_sorted[0].chapter_index}, para={paragraphs_sorted[0].paragraph_index}"
    )
    print(
        f"Last paragraph: book={paragraphs_sorted[-1].book_index}, chapter={paragraphs_sorted[-1].chapter_index}, para={paragraphs_sorted[-1].paragraph_index}"
    )

    return paragraphs_sorted, repos


@app.cell
def _(paragraphs_sorted):
    len(paragraphs_sorted)
    return


@app.cell
def _(mo):
    mo.md(r"""### Extract embeddings and calculate similarities""")
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
def _(mo):
    mo.md(
        r"""
    ### Get cummulative chapter, paragraph indices
    For plotting purposes - calculate position from beginning of book to extract specific chapters from cos_mat
    """
    )
    return


@app.cell
def _(Counter, repos):
    chapters = repos.chapters.list_all()
    chapter_sorted = sorted(chapters, key=lambda c: (c.book_index, c.chapter_index))
    chapter_books = [c.book_index for c in chapter_sorted]
    book_chapter_counts = Counter(chapter_books)
    print("Chapter counts per book:")
    for _book, _count in book_chapter_counts.items():
        print(f"Book {_book}: {_count} chapters")
    return (book_chapter_counts,)


@app.cell
def _(book_chapter_counts, np):
    book_chapter_cumsum = np.cumsum(
        [0] + [book_chapter_counts[k] for k in book_chapter_counts.keys()]
    )
    print("Cumulative chapters counts per book:")
    for _book, _cumsum in zip(book_chapter_counts.keys(), book_chapter_cumsum[1:]):
        print(f"Book {_book}: {_cumsum} cumulative chapters")
    return (book_chapter_cumsum,)


@app.cell
def _(Counter, paragraphs_sorted):
    paragraph_books = [p.book_index for p in paragraphs_sorted]
    # count paragraphs per book

    book_counts = Counter(paragraph_books)
    print("Paragraph counts per book:")
    for book, count in book_counts.items():
        print(f"Book {book}: {count} paragraphs")
    return (book_counts,)


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
    chapter_cumsum = np.cumsum(
        [0] + [chapter_counts[(k[0], k[1])] for k in chapter_counts.keys()]
    )
    print("Cumulative paragraph counts per chapter:")
    for (_book, _chapter), _cumsum in zip(chapter_counts.keys(), chapter_cumsum[1:]):
        print(f"Book {_book}, Chapter {_chapter}: {_cumsum} cumulative paragraphs")
    return (chapter_cumsum,)


@app.cell
def _(mo):
    mo.md(r"""### Plot full book similarity matrix""")
    return


@app.cell
def _(book_cumsum, chapter_cumsum, cm, colors, cos_mat, plt):
    # increase figure size for better visibility
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    ax.matshow(cos_mat, cmap="viridis")
    # add horizontal and vertical lines to demarcate books
    for _cumsum in book_cumsum[1:]:
        ax.axhline(y=_cumsum - 0.5, color="red", linestyle="--")
        ax.axvline(x=_cumsum - 0.5, color="red", linestyle="--")
    # add horizontal and vertical lines to demarcate chapters
    # use thin black lines for chapter boundaries
    for _cumsum in chapter_cumsum[1:]:
        ax.axhline(y=_cumsum - 0.5, color="red", linestyle=":", linewidth=1)
        ax.axvline(x=_cumsum - 0.5, color="red", linestyle=":", linewidth=1)
    ax.set_title("Cosine Similarity Matrix of Paragraph Vectors")
    ax.set_xlabel("Paragraph Index")
    ax.set_ylabel("Paragraph Index")
    cmap = plt.get_cmap("viridis")
    norm = colors.Normalize(0, 5)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    fig.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""### Plot similarity matrix for a specific chapter""")
    return


@app.cell
def _(book_chapter_cumsum, chapter_cumsum):
    # get absolute chapter position
    book_i = 3
    chapter_i = 4
    chapter_i_abs = book_chapter_cumsum[book_i - 1] + chapter_i
    # get paraphraph indices for chapter
    p_start = chapter_cumsum[chapter_i_abs]
    p_end = chapter_cumsum[chapter_i_abs + 1]
    [p_start, p_end]
    return p_end, p_start


@app.cell
def _(cm, cos_mat, p_end, p_start, plt):
    cos_mat_filtered = cos_mat[p_start:p_end, p_start:p_end]  # Book 3, Chapter 5
    # _fig = plt.figure(figsize=(7, 7))
    _fig = plt.figure()
    _ax = _fig.add_subplot(111)
    _ax.matshow(cos_mat_filtered, cmap="viridis")
    _ax.set_title("Cosine Similarity Matrix of Paragraph Vectors")
    _ax.set_xlabel("Paragraph Index")
    _ax.set_ylabel("Paragraph Index")
    _cmap = plt.get_cmap("viridis")
    # _norm = colors.Normalize(0, 1)
    _fig.colorbar(cm.ScalarMappable(norm=None, cmap=_cmap), ax=_ax)
    _fig.gca()
    return (cos_mat_filtered,)


@app.cell
def _(mo):
    mo.md(r"""#### Plot similarity between paraphraphs n and n+1, n+3, n+5""")
    return


@app.cell
def _(cos_mat_filtered, plt):
    # rememeber to name local variables with leading underscore (e.g. '_fig')
    # repeat the above, but with each plot in a different subplot
    plt.figure(figsize=(12, 12))
    for _i in range(1, 6, 2):
        _ax = plt.subplot(3, 1, _i // 2 + 1)
        _ax.plot(cos_mat_filtered.diagonal(_i), marker="o", label=f"Diagonal {_i}")
        _ax.set_title(f"Diagonal {_i} Similarity")
        _ax.set_xlabel("Consecutive Paragraph Index")
        _ax.set_ylabel("Cosine Similarity")
        _ax.grid()
        _ax.legend()
        # _ax.set_xticks(range(len(cos_mat_filtered)))
        # _ax.set_xticklabels(range(p_start + _i, p_end + _i), rotation=45)
    plt.tight_layout()
    plt.gca()

    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
