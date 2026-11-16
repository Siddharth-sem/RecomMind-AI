import os
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer


# ==========================================
# CONFIGURATION
# ==========================================

CATALOG_PATH = (
    "datasets/products/processed/electronics_catalog.csv"
)

FEATURES_PATH = (
    "models/semantic_product_features.csv"
)

EMBEDDINGS_PATH = (
    "models/product_embeddings.npy"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ==========================================
# LOAD PRODUCT CATALOGUE
# ==========================================

print("Loading product catalogue...")

products = pd.read_csv(CATALOG_PATH)

print("Total products:", len(products))


# ==========================================
# CLEAN TEXT FIELDS
# ==========================================

text_columns = [
    "title",
    "brand",
    "main_category",
    "categories",
    "description",
    "features"
]

for column in text_columns:

    products[column] = (
        products[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )


# ==========================================
# BUILD SEMANTIC TEXT
# ==========================================

print("\nBuilding semantic product text...")

products["semantic_text"] = (
    "Product: " + products["title"]
    + ". Brand: " + products["brand"]
    + ". Category: " + products["categories"]
    + ". Description: " + products["description"].str[:1000]
    + ". Features: " + products["features"].str[:500]
)

# Remove products without meaningful text
products = products[
    products["title"].str.len() > 0
].reset_index(drop=True)

print("Products with valid titles:", len(products))


# ==========================================
# LOAD SENTENCE TRANSFORMER
# ==========================================

print("\nLoading Sentence Transformer...")

model = SentenceTransformer(MODEL_NAME)


# ==========================================
# GENERATE EMBEDDINGS
# ==========================================

print("\nGenerating product embeddings...")

embeddings = model.encode(
    products["semantic_text"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("Embedding shape:", embeddings.shape)


# ==========================================
# SAVE OUTPUTS
# ==========================================

os.makedirs("models", exist_ok=True)

feature_columns = [
    "parent_asin",
    "title",
    "brand",
    "main_category",
    "categories",
    "description",
    "features",
    "price",
    "average_rating",
    "rating_number",
    "image_url",
    "semantic_text"
]

products[feature_columns].to_csv(
    FEATURES_PATH,
    index=False
)

np.save(
    EMBEDDINGS_PATH,
    embeddings
)


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n========== PRODUCT EMBEDDINGS ==========")

print("Products:", len(products))
print("Embedding dimensions:", embeddings.shape[1])

print("Features saved:", FEATURES_PATH)
print("Embeddings saved:", EMBEDDINGS_PATH)

print("\nProduct embedding generation completed!")