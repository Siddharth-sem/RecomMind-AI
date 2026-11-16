import os
import numpy as np
import pandas as pd

from functools import lru_cache
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


FEATURES_PATH = "models/semantic_product_features.csv"
EMBEDDINGS_PATH = "models/product_embeddings.npy"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ==========================================
# LOAD PRODUCT DATA
# ==========================================

products = pd.read_csv(FEATURES_PATH)

embeddings = np.load(EMBEDDINGS_PATH)

print("Product semantic engine loaded!")
print("Products:", len(products))
print("Embeddings:", embeddings.shape)


# ==========================================
# LOAD MODEL
# ==========================================

@lru_cache(maxsize=1)
def load_model():

    return SentenceTransformer(MODEL_NAME)


# ==========================================
# SEMANTIC PRODUCT SEARCH
# ==========================================

def semantic_product_search(query, top_n=10):

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if top_n <= 0:
        raise ValueError("top_n must be positive.")

    model = load_model()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        query_embedding,
        embeddings
    ).flatten()

    indices = np.argsort(scores)[::-1][:top_n]

    results = products.iloc[indices].copy()

    results["semantic_similarity"] = scores[indices]

    return results.reset_index(drop=True)


# ==========================================
# SIMILAR PRODUCT SEARCH
# ==========================================

def get_similar_products(product_title, top_n=10):

    matches = products[
        products["title"].str.strip().str.lower()
        == product_title.strip().lower()
    ]

    if matches.empty:
        raise ValueError("Product not found.")

    index = matches.index[0]

    scores = cosine_similarity(
        embeddings[index].reshape(1, -1),
        embeddings
    ).flatten()

    # Exclude selected product
    scores[index] = -1

    indices = np.argsort(scores)[::-1]

    indices = [
        i for i in indices
        if scores[i] >= 0
    ][:top_n]

    results = products.iloc[indices].copy()

    results["semantic_similarity"] = scores[indices]

    return results.reset_index(drop=True)