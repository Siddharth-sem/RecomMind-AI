import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BOOK_DIR = os.path.join(BASE_DIR, "datasets", "books")


def load_book_data():

    print("Loading Book-Crossing datasets...")

    books = pd.read_csv(
        os.path.join(BOOK_DIR, "BX-Books.csv"),
        sep=";",
        encoding="latin-1",
        on_bad_lines="warn",
        low_memory=False
    )

    ratings = pd.read_csv(
        os.path.join(BOOK_DIR, "BX-Book-Ratings.csv"),
        sep=";",
        encoding="latin-1"
    )

    users = pd.read_csv(
        os.path.join(BOOK_DIR, "BX-Users.csv"),
        sep=";",
        encoding="latin-1"
    )

    return books, ratings, users


def clean_book_data():

    books, ratings, users = load_book_data()

    print("\nOriginal dataset:")
    print("Books:", len(books))
    print("Ratings:", len(ratings))
    print("Users:", len(users))

    # Clean column names
    books.columns = books.columns.str.strip()
    ratings.columns = ratings.columns.str.strip()
    users.columns = users.columns.str.strip()

    # Remove duplicate books
    books = books.drop_duplicates(subset="ISBN")

    # Remove books without titles
    books = books.dropna(subset=["Book-Title", "Book-Author"])

    # Clean titles and authors
    books["Book-Title"] = books["Book-Title"].astype(str).str.strip()
    books["Book-Author"] = books["Book-Author"].astype(str).str.strip()

    # Clean publication year
    books["Year-Of-Publication"] = pd.to_numeric(
        books["Year-Of-Publication"],
        errors="coerce"
    )

    books = books[
        books["Year-Of-Publication"].between(1900, 2026)
    ]

    # Keep only explicit ratings (1-10)
    ratings["Book-Rating"] = pd.to_numeric(
        ratings["Book-Rating"],
        errors="coerce"
    )

    ratings = ratings[
        ratings["Book-Rating"].between(1, 10)
    ]

    # Remove ratings for books not present in metadata
    ratings = ratings[
        ratings["ISBN"].isin(books["ISBN"])
    ]

    # Keep users with at least 5 ratings
    user_counts = ratings["User-ID"].value_counts()

    ratings = ratings[
        ratings["User-ID"].isin(
            user_counts[user_counts >= 5].index
        )
    ]

    # Keep books with at least 5 ratings
    book_counts = ratings["ISBN"].value_counts()

    ratings = ratings[
        ratings["ISBN"].isin(
            book_counts[book_counts >= 5].index
        )
    ]

    # Keep metadata only for books with ratings
    books = books[
        books["ISBN"].isin(ratings["ISBN"])
    ]

    # Save cleaned datasets
    books.to_csv(
        os.path.join(BOOK_DIR, "clean_books.csv"),
        index=False
    )

    ratings.to_csv(
        os.path.join(BOOK_DIR, "clean_ratings.csv"),
        index=False
    )

    print("\nCleaning completed!")
    print("Clean books:", len(books))
    print("Clean ratings:", len(ratings))
    print("Unique users:", ratings["User-ID"].nunique())
    print("Unique books:", ratings["ISBN"].nunique())

    return books, ratings


if __name__ == "__main__":
    clean_book_data()