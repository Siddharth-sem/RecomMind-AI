import os
import json
import pandas as pd


# ==========================================
# PATHS
# ==========================================

REVIEWS_PATH = (
    "datasets/products/processed/electronics_reviews.csv"
)

METADATA_PATH = (
    "datasets/products/raw/electronics_metadata.jsonl"
)

OUTPUT_PATH = (
    "datasets/products/processed/electronics_catalog.csv"
)


# ==========================================
# LOAD SELECTED PRODUCT IDS
# ==========================================

print("Loading selected product IDs...")

reviews = pd.read_csv(
    REVIEWS_PATH,
    usecols=["parent_asin"],
    dtype={"parent_asin": str}
)

selected_products = set(
    reviews["parent_asin"].dropna().unique()
)

print("Selected products:", len(selected_products))


# ==========================================
# EXTRACT PRODUCT METADATA
# ==========================================

print("\nScanning metadata file...")
print("This may take some time because the file is 5.25 GB.")

products = {}

processed = 0

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as file:

    for line in file:

        processed += 1

        try:
            item = json.loads(line)

        except json.JSONDecodeError:
            continue

        asin = item.get("parent_asin")

        if asin not in selected_products:
            continue

        # Extract product image
        images = item.get("images") or []

        image_url = None

        if images:

            first_image = images[0]

            image_url = (
                first_image.get("hi_res")
                or first_image.get("large")
                or first_image.get("thumb")
            )

        # Extract description
        description = item.get("description") or []

        if isinstance(description, list):
            description = " ".join(
                str(x) for x in description
            )

        # Extract features
        features = item.get("features") or []

        if isinstance(features, list):
            features = " ".join(
                str(x) for x in features
            )

        # Extract categories
        categories = item.get("categories") or []

        if isinstance(categories, list):
            categories = " > ".join(
                str(x) for x in categories
            )

        # Extract manufacturer
        details = item.get("details") or {}

        manufacturer = None

        if isinstance(details, dict):
            manufacturer = details.get("Manufacturer")

        products[asin] = {

            "parent_asin": asin,

            "title": item.get("title"),

            "brand": (
                manufacturer
                or item.get("store")
            ),

            "store": item.get("store"),

            "main_category": item.get("main_category"),

            "categories": categories,

            "description": description,

            "features": features,

            "price": item.get("price"),

            "average_rating": item.get("average_rating"),

            "rating_number": item.get("rating_number"),

            "image_url": image_url
        }

        if len(products) == len(selected_products):
            print("\nAll selected products found!")
            break

        if processed % 500000 == 0:

            print(
                f"Scanned {processed:,} records | "
                f"Found {len(products):,} products"
            )


# ==========================================
# SAVE CATALOGUE
# ==========================================

print("\nSaving product catalogue...")

catalogue = pd.DataFrame(
    list(products.values())
)

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

catalogue.to_csv(
    OUTPUT_PATH,
    index=False
)


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n========== PRODUCT CATALOGUE ==========")

print("Products selected:", len(selected_products))

print("Products found:", len(catalogue))

print(
    "Products missing:",
    len(selected_products) - len(catalogue)
)

print(
    "Descriptions available:",
    catalogue["description"].notna().sum()
)

print(
    "Images available:",
    catalogue["image_url"].notna().sum()
)

print(
    "Prices available:",
    catalogue["price"].notna().sum()
)

print("\nCatalogue columns:")
print(catalogue.columns.tolist())

print("\nSample products:")

print(
    catalogue[
        ["title", "brand", "price", "average_rating"]
    ].head(10).to_string(index=False)
)

print("\nSaved to:", OUTPUT_PATH)