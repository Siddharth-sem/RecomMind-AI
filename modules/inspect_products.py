import pandas as pd

path = "datasets/products/raw/electronics_reviews.csv"

total_rows = 0
users = set()
products = set()
ratings_count = {}

for chunk in pd.read_csv(path, chunksize=200000):

    total_rows += len(chunk)

    users.update(chunk["user_id"].unique())
    products.update(chunk["parent_asin"].unique())

    counts = chunk["rating"].value_counts()

    for rating, count in counts.items():
        ratings_count[rating] = (
            ratings_count.get(rating, 0) + count
        )

    print("Processed:", total_rows)

print("\n========== DATASET SUMMARY ==========")
print("Total interactions:", total_rows)
print("Unique users:", len(users))
print("Unique products:", len(products))
print("Rating distribution:", ratings_count)