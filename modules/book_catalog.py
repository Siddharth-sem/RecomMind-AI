import os
import ast
import re
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BOOK_DIR = os.path.join(BASE_DIR, "datasets", "books")

CLEAN_BOOKS = os.path.join(BOOK_DIR, "clean_books.csv")
GOOGLE_BOOKS = os.path.join(BOOK_DIR, "google_books_catalog.csv")

OUTPUT_FILE = os.path.join(
    BOOK_DIR,
    "unified_book_catalog.csv"
)


# ==========================================
# ISBN NORMALIZATION
# ==========================================

def normalize_isbn(isbn):

    if pd.isna(isbn):
        return None

    isbn = re.sub(r"[^0-9Xx]", "", str(isbn)).upper()

    # ISBN-10: first 9 characters must be digits.
    # The final character may be a digit or X.
    if re.fullmatch(r"\d{9}[\dX]", isbn):

        isbn13 = "978" + isbn[:9]

        checksum = sum(
            int(digit) * (1 if i % 2 == 0 else 3)
            for i, digit in enumerate(isbn13)
        )

        check_digit = (10 - checksum % 10) % 10

        return isbn13 + str(check_digit)

    # ISBN-13 must contain exactly 13 digits.
    if re.fullmatch(r"\d{13}", isbn):
        return isbn

    return None

# ==========================================
# LOAD DATA
# ==========================================

def build_catalog():

    books = pd.read_csv(
        CLEAN_BOOKS,
        dtype={"ISBN": str}
    )

    google = pd.read_csv(
        GOOGLE_BOOKS
    )

    books["isbn_key"] = books["ISBN"].apply(normalize_isbn)

    google["industry_identifiers"] = (
        google["industry_identifiers"]
        .fillna("[]")
        .apply(ast.literal_eval)
    )

    def get_google_isbn(identifiers):

        for identifier in identifiers:

            value = normalize_isbn(
                identifier.get("identifier")
            )

            if value:
                return value

        return None

    google["isbn_key"] = (
        google["industry_identifiers"]
        .apply(get_google_isbn)
    )

    # ======================================
    # ENRICH BOOK-CROSSING BOOKS
    # ======================================

    google_metadata = google[
        [
            "isbn_key",
            "description",
            "categories",
            "thumbnail",
            "preview_link",
            "published_date"
        ]
    ].dropna(subset=["isbn_key"])

    google_metadata = google_metadata.drop_duplicates(
        subset="isbn_key"
    )

    enriched = books.merge(
        google_metadata,
        on="isbn_key",
        how="left"
    )

    enriched["source"] = "Book-Crossing"

    # ======================================
    # ADD GOOGLE-ONLY BOOKS
    # ======================================

    matched_isbns = set(
        enriched["isbn_key"].dropna()
    )

    google_only = google[
        google["isbn_key"].notna()
        & ~google["isbn_key"].isin(matched_isbns)
    ].copy()

    google_only_books = pd.DataFrame({

        "ISBN": "GOOGLE:" + google_only["google_id"].astype(str),

        "Book-Title": google_only["title"],

        "Book-Author": google_only["authors"],

        "Year-Of-Publication": pd.to_numeric(
            google_only["published_date"].str[:4],
            errors="coerce"
        ),

        "Publisher": google_only["publisher"],

        "Image-URL-S": google_only["thumbnail"],

        "Image-URL-M": google_only["thumbnail"],

        "Image-URL-L": google_only["thumbnail"],

        "isbn_key": google_only["isbn_key"],

        "description": google_only["description"],

        "categories": google_only["categories"],

        "thumbnail": google_only["thumbnail"],

        "preview_link": google_only["preview_link"],

        "published_date": google_only["published_date"],

        "source": "Google Books"

    })

    # ======================================
    # FINAL CATALOGUE
    # ======================================

    unified = pd.concat(
        [enriched, google_only_books],
        ignore_index=True
    )

    unified["description"] = unified["description"].fillna("")
    unified["categories"] = unified["categories"].fillna("")
    unified["thumbnail"] = unified["thumbnail"].fillna("")

    unified.drop_duplicates(
        subset=["ISBN"],
        inplace=True
    )

    unified.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print("\n========== UNIFIED CATALOGUE ==========")

    print("Book-Crossing books:", len(books))
    print("Google Books entries:", len(google))
    print("ISBN-matched Google entries:",
          enriched["description"].notna().sum())

    print("Google-only books:", len(google_only_books))

    print("Final catalogue:", len(unified))

    print("\nDescription coverage:",
          round((unified["description"] != "").mean() * 100, 2), "%")

    print("\nSources:")
    print(unified["source"].value_counts())

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    build_catalog()