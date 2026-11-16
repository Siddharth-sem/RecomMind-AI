import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BOOK_DIR = os.path.join(BASE_DIR, "datasets", "books")


# ==========================================
# LOAD CLEANED DATA
# ==========================================

books = pd.read_csv(
    os.path.join(BOOK_DIR, "clean_books.csv")
)

ratings = pd.read_csv(
    os.path.join(BOOK_DIR, "clean_ratings.csv")
)

books["Book-Title"] = books["Book-Title"].fillna("")
books["Book-Author"] = books["Book-Author"].fillna("")
books["Publisher"] = books["Publisher"].fillna("")

books["features"] = (
    books["Book-Title"] + " " +
    books["Book-Author"] + " " +
    books["Publisher"]
)


# ==========================================
# CONTENT-BASED MODEL
# ==========================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=20000
)

book_vectors = vectorizer.fit_transform(
    books["features"]
)


# ==========================================
# COLLABORATIVE FILTERING
# ==========================================

rating_matrix = ratings.pivot_table(
    index="User-ID",
    columns="ISBN",
    values="Book-Rating",
    fill_value=0
)

svd = TruncatedSVD(
    n_components=50,
    random_state=42
)

user_factors = svd.fit_transform(rating_matrix)

book_factors = svd.components_


# ==========================================
# CONTENT RECOMMENDATIONS
# ==========================================

def get_similar_books(book_title, top_n=10):

    matches = books[
        books["Book-Title"].str.strip().str.lower()
        == book_title.strip().lower()
    ]

    if matches.empty:
        raise ValueError("Book not found.")

    index = matches.index[0]

    scores = cosine_similarity(
        book_vectors[index],
        book_vectors
    ).flatten()

    # Exclude all editions of the selected title
    same_title = (
        books["Book-Title"].str.strip().str.lower()
        == book_title.strip().lower()
    )

    scores[same_title.values] = -1

    indices = np.argsort(scores)[::-1]

    indices = [
        i for i in indices
        if scores[i] >= 0
    ][:top_n]

    results = books.iloc[indices].copy()
    results["similarity"] = scores[indices]

    return results.reset_index(drop=True)


# ==========================================
# COLLABORATIVE RECOMMENDATIONS
# ==========================================

def get_collaborative_book_recommendations(
    user_id,
    top_n=10
):

    if user_id not in rating_matrix.index:
        raise ValueError("User ID not found.")

    user_index = rating_matrix.index.get_loc(user_id)

    predictions = np.dot(
        user_factors[user_index],
        book_factors
    )

    recommendations = pd.DataFrame({
        "ISBN": rating_matrix.columns,
        "predicted_rating": predictions
    })

    already_rated = ratings[
        ratings["User-ID"] == user_id
    ]["ISBN"].tolist()

    recommendations = recommendations[
        ~recommendations["ISBN"].isin(already_rated)
    ]

    recommendations = recommendations.merge(
        books,
        on="ISBN",
        how="left"
    )

    return recommendations.sort_values(
        "predicted_rating",
        ascending=False
    ).head(top_n).reset_index(drop=True)


print("Book recommendation engine loaded!")
print("Total books:", len(books))
print("Total ratings:", len(ratings))
print("Rating matrix:", rating_matrix.shape)