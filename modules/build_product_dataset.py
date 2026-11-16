import os
import pandas as pd

RAW_PATH = "datasets/products/raw/electronics_reviews.csv"
OUTPUT_DIR = "datasets/products/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CHUNK_SIZE = 200000
TOP_PRODUCTS = 2000
MIN_INTERACTIONS = 5


# ==========================================
# STEP 1: COUNT PRODUCT INTERACTIONS
# ==========================================

print("Step 1: Counting product interactions...")

product_counts = {}

for chunk in pd.read_csv(
    RAW_PATH,
    chunksize=CHUNK_SIZE
):

    counts = chunk["parent_asin"].value_counts()

    for asin, count in counts.items():

        product_counts[asin] = (
            product_counts.get(asin, 0) + count
        )

print("Unique products:", len(product_counts))


# ==========================================
# STEP 2: SELECT POPULAR PRODUCTS
# ==========================================

print("\nStep 2: Selecting popular products...")

popular_products = sorted(
    product_counts,
    key=product_counts.get,
    reverse=True
)[:TOP_PRODUCTS]

popular_products = set(popular_products)

print("Selected products:", len(popular_products))


# ==========================================
# STEP 3: EXTRACT THEIR INTERACTIONS
# ==========================================

print("\nStep 3: Extracting interactions...")

selected_chunks = []

for chunk in pd.read_csv(
    RAW_PATH,
    chunksize=CHUNK_SIZE
):

    filtered = chunk[
        chunk["parent_asin"].isin(popular_products)
    ]

    if not filtered.empty:
        selected_chunks.append(filtered)

data = pd.concat(
    selected_chunks,
    ignore_index=True
)

print("Extracted interactions:", len(data))


# ==========================================
# STEP 4: ITERATIVE 5-CORE FILTERING
# ==========================================

print("\nStep 4: Applying 5-core filtering...")

previous_size = -1

while previous_size != len(data):

    previous_size = len(data)

    user_counts = data["user_id"].value_counts()
    product_counts = data["parent_asin"].value_counts()

    valid_users = user_counts[
        user_counts >= MIN_INTERACTIONS
    ].index

    valid_products = product_counts[
        product_counts >= MIN_INTERACTIONS
    ].index

    data = data[
        data["user_id"].isin(valid_users)
        & data["parent_asin"].isin(valid_products)
    ].copy()

    print("Remaining interactions:", len(data))


# ==========================================
# STEP 5: SAVE CLEAN DATASET
# ==========================================

data = data.drop_duplicates(
    subset=["user_id", "parent_asin"]
)

output_path = os.path.join(
    OUTPUT_DIR,
    "electronics_reviews.csv"
)

data.to_csv(
    output_path,
    index=False
)


# ==========================================
# STEP 6: FINAL SUMMARY
# ==========================================

print("\n========== FINAL DATASET ==========")

print("Interactions:", len(data))
print("Unique users:", data["user_id"].nunique())
print("Unique products:", data["parent_asin"].nunique())

print("\nRating distribution:")
print(data["rating"].value_counts().sort_index())

print("\nSaved to:", output_path)