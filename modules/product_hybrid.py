import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from modules.product_recommender import (
    collaborative_product_recommendations,
    user_product_matrix
)

from modules.product_semantic import semantic_product_search


def hybrid_product_recommendations(
    user_id,
    query,
    top_n=10,
    collaborative_weight=0.4,
    semantic_weight=0.6
):

    if user_id not in user_product_matrix.index:
        raise ValueError("User not found.")

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    total_weight = collaborative_weight + semantic_weight

    if total_weight <= 0:
        raise ValueError("Weights must have a positive sum.")

    collaborative_weight /= total_weight
    semantic_weight /= total_weight

    # Semantic candidate generation
    semantic_results = semantic_product_search(
        query, top_n=100
    )

    # Collaborative recommendations
    cf_results = collaborative_product_recommendations(
        user_id, top_n=200
    )

    # Keep semantic candidates as the main candidate pool
    combined = semantic_results[
        ["parent_asin", "semantic_similarity"]
    ].copy()

    cf_scores = cf_results[
        ["parent_asin", "collaborative_score"]
    ].copy()

    # Attach collaborative scores where available
    combined = combined.merge(
        cf_scores,
        on="parent_asin",
        how="left"
    )

    # Missing CF scores get the lowest collaborative baseline
    combined["collaborative_score"] = (
        combined["collaborative_score"].fillna(0)
    )

    # Normalize both score columns
    for column, normalized_column in [
        ("semantic_similarity", "semantic_normalized"),
        ("collaborative_score", "collaborative_normalized")
    ]:

        values = combined[column].values.reshape(-1, 1)

        if values.max() == values.min():
            combined[normalized_column] = 0.5
        else:
            combined[normalized_column] = (
                MinMaxScaler().fit_transform(values).flatten()
            )

    # Hybrid ranking
    combined["hybrid_score"] = (
        combined["semantic_normalized"] * semantic_weight
        + combined["collaborative_normalized"] * collaborative_weight
    )

    # Exclude previously rated products
    rated_products = set(
        user_product_matrix.columns[
            user_product_matrix.loc[user_id] > 0
        ]
    )

    combined = combined[
        ~combined["parent_asin"].isin(rated_products)
    ]

    # Attach product metadata
    catalog = pd.read_csv(
        "datasets/products/processed/electronics_catalog.csv"
    )

    combined = combined.merge(
        catalog,
        on="parent_asin",
        how="left"
    )

    combined = combined.sort_values(
        "hybrid_score",
        ascending=False
    )

    return combined.head(top_n).reset_index(drop=True)