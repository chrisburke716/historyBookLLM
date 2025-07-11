import marimo

__generated_with = "0.14.10"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import weaviate
    from weaviate.classes.query import Sort
    import numpy as np

    from history_book.data_models.book import (
        ParagraphDBModel,
    )
    return Sort, np, weaviate


@app.cell
def _(Sort, weaviate):
    client = weaviate.connect_to_local()
    paragraph_collection = client.collections.get("Paragraphs")

    paragraphs = paragraph_collection.query.fetch_objects(
        sort=Sort.by_property(name="book_index")
        .by_property(name="chapter_index")
        .by_property(name="paragraph_index"),
        include_vector=True,
    )
    return (paragraphs,)


@app.cell
def _(paragraphs):
    type(paragraphs)
    return


@app.cell
def _(paragraphs):
    len(paragraphs.objects)
    return


@app.cell
def _(np, paragraphs):
    vec_list = [np.array(obj.vector['text_vector']) for obj in paragraphs.objects]
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
    cos_mat = np.dot(vec_mat, vec_mat.T)
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
