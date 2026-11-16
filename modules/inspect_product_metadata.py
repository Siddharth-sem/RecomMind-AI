import json

path = "datasets/products/raw/electronics_metadata.jsonl"

with open(path, "r", encoding="utf-8") as file:
    record = json.loads(next(file))

print("Available fields:")
print(list(record.keys()))

print("\nSample product information:")

for key in [
    "parent_asin",
    "title",
    "main_category",
    "average_rating",
    "rating_number",
    "price",
    "store",
    "categories",
    "features",
    "description",
    "details",
    "images"
]:
    value = record.get(key)

    # Avoid printing huge image or description data
    print(f"\n{key}:")
    print(str(value)[:400])