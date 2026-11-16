import os
import numpy as np
import pandas as pd

from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BOOK_DIR = os.path.join(BASE_DIR, "datasets", "books")

MODEL_DIR = os.path.join(BASE_DIR, "models")

CATALOG_PATH = os.path.join(
    BOOK_DIR,
    "unified_book_catalog.csv"
)

EMBEDDING_PATH = os.path.join(
    MODEL_DIR,
    "book_embeddings.npy"
)

FEATURE_PATH = os.path.join(
    MODEL_DIR,
    "semantic_book_features.csv"
)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ==========================================
# LOAD CATALOGUE
# ==========================================

books = pd.read_csv(CATALOG_PATH)

for column in [
    "Book-Title",
    "Book-Author",
    "Publisher",
    "description",
    "categories"
]:
    books[column] = books[column].fillna("").astype(str)


# ==========================================
# CREATE TEXT FEATURES
# ==========================================

books["semantic_text"] = (
    "Title: " + books["Book-Title"] + ". "
    "Author: " + books["Book-Author"] + ". "
    "Publisher: " + books["Publisher"] + ". "
    "Categories: " + books["categories"] + ". "
    "Description: " + books["description"]
)

books["semantic_text"] = (
    books["semantic_text"].str.strip()
)


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

@lru_cache(maxsize=1)
def get_embedding_model():

    from sentence_transformers import SentenceTransformer

    print("Loading book embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    print("Embedding model loaded!")

    return model


# ==========================================
# GENERATE EMBEDDINGS
# ==========================================

def build_book_embeddings():

    model = get_embedding_model()

    print("Generating book embeddings...")

    embeddings = model.encode(
        books["semantic_text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    os.makedirs(MODEL_DIR, exist_ok=True)

    np.save(
        EMBEDDING_PATH,
        embeddings
    )

    feature_data = books[
        [
            "ISBN",
            "Book-Title",
            "Book-Author",
            "Publisher",
            "categories",
            "description",
            "source"
        ]
    ].copy()

    feature_data.to_csv(
        FEATURE_PATH,
        index=False
    )

    print("\nBook embeddings generated!")
    print("Shape:", embeddings.shape)


if __name__ == "__main__":
    build_book_embeddings()

    # ==========================================
# LOAD SAVED EMBEDDINGS
# ==========================================

@lru_cache(maxsize=1)
def load_book_embeddings():

    return np.load(EMBEDDING_PATH)


# ==========================================
# SEMANTIC SEARCH
# ==========================================

def semantic_book_search(query, top_n=10):

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Enter a valid search query.")

    if not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n must be positive.")

    model = get_embedding_model()
    embeddings = load_book_embeddings()

    query_embedding = model.encode(
        [query.strip()],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    results = books.copy()

    results["semantic_similarity"] = scores

    return results.sort_values(
        "semantic_similarity",
        ascending=False
    ).head(top_n).reset_index(drop=True)


# ==========================================
# SIMILAR BOOK RECOMMENDATIONS
# ==========================================

def get_semantically_similar_books(book_title, top_n=10):

    if not isinstance(book_title, str) or not book_title.strip():
        raise ValueError("Enter a valid book title.")

    selected_title = book_title.strip().lower()

    matches = books[
        books["Book-Title"].str.strip().str.lower()
        == selected_title
    ]

    if matches.empty:
        raise ValueError("Book not found.")

    index = matches.index[0]

    embeddings = load_book_embeddings()

    scores = cosine_similarity(
        embeddings[index:index + 1],
        embeddings
    )[0]

    results = books.copy()

    results["semantic_similarity"] = scores

    # Normalize titles to identify duplicate editions
    results["normalized_title"] = (
        results["Book-Title"]
        .str.strip()
        .str.lower()
    )

    # Exclude every edition of the selected book
    results = results[
        results["normalized_title"] != selected_title
    ]

    # Keep only the highest-scoring edition of each title
    results = (
        results.sort_values(
            "semantic_similarity",
            ascending=False
        )
        .drop_duplicates(
            subset=["normalized_title"],
            keep="first"
        )
    )

    return results.head(top_n).drop(
        columns=["normalized_title"]
    ).reset_index(drop=True)