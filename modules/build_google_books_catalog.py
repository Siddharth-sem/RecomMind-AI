import os
import time
import pandas as pd

from modules.google_books import search_books


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "books",
    "google_books_catalog.csv"
)


SEARCH_QUERIES = [
    "fiction",
    "science fiction",
    "psychological thriller",
    "fantasy novels",
    "romance novels",
    "self help",
    "technology and computer science",
    "business and economics",
    "history",
    "mystery novels"
]


def build_catalog():

    all_books = []

    for query in SEARCH_QUERIES:

        print(f"\nSearching: {query}")

        results = search_books(query, max_results=20)

        all_books.extend(results)

        print(f"Retrieved: {len(results)} books")

        # Avoid sending requests too quickly
        time.sleep(1)

    if not all_books:
        print("No books retrieved. Check API access.")
        return

    df = pd.DataFrame(all_books)

    # Remove duplicate Google Books entries
    df.drop_duplicates(
        subset="google_id",
        inplace=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    print("\n========== CATALOGUE COMPLETE ==========")
    print("Total unique books:", len(df))
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    build_catalog()