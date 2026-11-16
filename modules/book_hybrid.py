import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from modules.book_recommender import (
    get_collaborative_book_recommendations
)

from modules.book_semantic import (
    get_semantically_similar_books
)


def hybrid_book_recommendations(
    user_id,
    book_title,
    top_n=10,
    collaborative_weight=0.6,
    semantic_weight=0.4
):
    # Get collaborative recommendations
    collaborative = get_collaborative_book_recommendations(
        user_id,
        top_n=50
    )

    # Get semantic recommendations
    semantic = get_semantically_similar_books(
        book_title,
        top_n=50
    )

    # Keep only required columns
    collaborative = collaborative[
        ["Book-Title", "Book-Author", "predicted_rating"]
    ].copy()

    semantic = semantic[
        ["Book-Title", "Book-Author", "semantic_similarity"]
    ].copy()

    # Normalize titles for matching
    collaborative["key"] = (
        collaborative["Book-Title"].str.strip().str.lower()
    )

    semantic["key"] = (
        semantic["Book-Title"].str.strip().str.lower()
    )

    # Merge both recommendation sources
    hybrid = pd.merge(
        collaborative,
        semantic,
        on="key",
        how="outer",
        suffixes=("_cf", "_semantic")
    )

    # Combine title and author columns
    hybrid["Book-Title"] = (
        hybrid["Book-Title_cf"]
        .fillna(hybrid["Book-Title_semantic"])
    )

    hybrid["Book-Author"] = (
        hybrid["Book-Author_cf"]
        .fillna(hybrid["Book-Author_semantic"])
    )

    # Fill missing scores
    hybrid["predicted_rating"] = hybrid[
        "predicted_rating"
    ].fillna(0)

    hybrid["semantic_similarity"] = hybrid[
        "semantic_similarity"
    ].fillna(0)

    # Normalize scores independently
    scaler = MinMaxScaler()

    hybrid[["cf_score", "semantic_score"]] = scaler.fit_transform(
        hybrid[["predicted_rating", "semantic_similarity"]]
    )

    # Calculate hybrid score
    hybrid["hybrid_score"] = (
        collaborative_weight * hybrid["cf_score"]
        + semantic_weight * hybrid["semantic_score"]
    )

    # Sort recommendations
    hybrid = hybrid.sort_values(
        "hybrid_score",
        ascending=False
    )

    return hybrid[
        [
            "Book-Title",
            "Book-Author",
            "predicted_rating",
            "semantic_similarity",
            "hybrid_score"
        ]
    ].head(top_n).reset_index(drop=True)