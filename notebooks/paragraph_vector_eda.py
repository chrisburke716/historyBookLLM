import marimo

__generated_with = "0.14.10"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import weaviate
    from weaviate.classes.query import Sort
    import numpy as np

    # Use the fixed repository pattern instead of direct client
    from history_book.database.repositories import BookRepositoryManager
    from history_book.database.config import WeaviateConfig
    return Sort, np, weaviate, BookRepositoryManager, WeaviateConfig


@app.cell
def _(BookRepositoryManager, WeaviateConfig, Sort):
    # Use the repository pattern with the fix
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
    
    return paragraphs_sorted, repos


@app.cell
def _(repos):
    # Clean up connections at the end of the notebook
    repos.close_all()
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
def _(weaviate, np, paragraphs_sorted):
    # Get vectors using direct weaviate connection for now
    # TODO: Add vector support to repository pattern
    
    client = weaviate.connect_to_local()
    paragraph_collection = client.collections.get("Paragraphs")
    
    # Get ALL paragraphs with vectors in one query
    print("Fetching all paragraphs with vectors...")
    result = paragraph_collection.query.fetch_objects(
        include_vector=True,
        limit=10000  # Get all paragraphs (we have 3419)
    )
    
    # Extract vectors and match them to our sorted paragraphs by ID
    vector_by_id = {}
    for obj in result.objects:
        if obj.vector and 'text_vector' in obj.vector:
            vector_by_id[str(obj.uuid)] = np.array(obj.vector['text_vector'])
    
    # Create ordered vector list matching paragraphs_sorted
    vec_list = []
    missing_count = 0
    for paragraph in paragraphs_sorted:
        if paragraph.id in vector_by_id:
            vec_list.append(vector_by_id[paragraph.id])
        else:
            missing_count += 1
    
    client.close()
    print(f"Extracted {len(vec_list)} vectors from {len(paragraphs_sorted)} paragraphs")
    if missing_count > 0:
        print(f"Warning: {missing_count} paragraphs missing vectors")
    return (vec_list,)


@app.cell
def _(np, vec_list):
    vec_mat = np.stack(vec_list)
    return (vec_mat,)


@app.cell
def _(vec_mat):
    vec_mat.shape
    return


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
def _(cos_mat, plt):
    plt.matshow(cos_mat)
    return


@app.cell
def _():
    # todo: add book, chapter delimeters to above
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
