import os
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "models")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ==========================================
# LOAD DATASETS
# ==========================================

ratings = pd.read_csv(
    os.path.join(DATASET_DIR, "movie_ratings.csv")
)

movie_features = pd.read_csv(
    os.path.join(MODEL_DIR, "movie_features.csv")
)

semantic_features = pd.read_csv(
    os.path.join(MODEL_DIR, "semantic_movie_features.csv")
)

movie_embeddings = np.load(
    os.path.join(MODEL_DIR, "movie_embeddings.npy")
)


# ==========================================
# LOAD TRAINED MODELS
# ==========================================

content_similarity = joblib.load(
    os.path.join(MODEL_DIR, "cosine_similarity.pkl")
)

svd_predictions = joblib.load(
    os.path.join(MODEL_DIR, "improved_svd_predictions.pkl")
)

user_means = joblib.load(
    os.path.join(MODEL_DIR, "user_means.pkl")
)

improved_svd_movie_ids = np.load(
    os.path.join(MODEL_DIR, "improved_svd_movie_ids.npy")
)

print("RecomMind movie engine loaded successfully!")


# ==========================================
# LAZY LOAD EMBEDDING MODEL
# ==========================================

@lru_cache(maxsize=1)
def get_embedding_model():

    from sentence_transformers import SentenceTransformer

    print("Loading semantic embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Semantic embedding model loaded!")

    return model


# ==========================================
# INPUT VALIDATION
# ==========================================

def validate_user(user_id):

    if isinstance(user_id, bool) or not isinstance(user_id, (int, np.integer)):
        raise ValueError("User ID must be an integer.")

    if user_id not in user_means.index:
        raise ValueError(f"User ID {user_id} does not exist.")

    return int(user_id)


def validate_top_n(top_n):

    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n must be a positive integer.")


def validate_query(query):

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Please enter a valid recommendation query.")

    return query.strip()


# ==========================================
# 1. COLLABORATIVE RECOMMENDATIONS
# ==========================================

def get_collaborative_recommendations(user_id, top_n=10):

    user_id = validate_user(user_id)
    validate_top_n(top_n)

    user_index = user_means.index.get_loc(user_id)

    predictions = np.asarray(
        svd_predictions[user_index]
    ).reshape(-1)

    if len(predictions) != len(improved_svd_movie_ids):
        raise ValueError("SVD predictions and movie ID mapping do not match.")

    recommendations = pd.DataFrame({
        "movie_id": improved_svd_movie_ids,
        "predicted_rating": predictions
    })

    # Exclude previously rated movies
    watched_movies = ratings.loc[
        ratings["user_id"] == user_id,
        "movie_id"
    ].unique()

    recommendations = recommendations[
        ~recommendations["movie_id"].isin(watched_movies)
    ]

    recommendations = recommendations.sort_values(
        "predicted_rating",
        ascending=False
    )

    recommendations = recommendations.merge(
        movie_features[
            ["movie_id", "title", "genre_text"]
        ],
        on="movie_id",
        how="left"
    )

    return recommendations.head(top_n).reset_index(drop=True)


# ==========================================
# 2. SEMANTIC SEARCH
# ==========================================

def get_semantic_scores(query):

    query = validate_query(query)

    embedding_model = get_embedding_model()

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    similarities = cosine_similarity(
        query_embedding,
        movie_embeddings
    )[0]

    results = semantic_features[
        ["movie_id"]
    ].copy()

    results["semantic_similarity"] = similarities

    return results.sort_values(
        "semantic_similarity",
        ascending=False
    ).reset_index(drop=True)


# ==========================================
# 3. CONTENT-BASED RECOMMENDATIONS
# ==========================================

def get_content_recommendations(movie_title, top_n=10):

    validate_top_n(top_n)

    if not isinstance(movie_title, str) or not movie_title.strip():
        raise ValueError("Enter a valid movie title.")

    matches = movie_features[
        movie_features["title"].str.lower()
        == movie_title.strip().lower()
    ]

    if matches.empty:
        raise ValueError(f"Movie not found: {movie_title}")

    movie_index = matches.index[0]

    similarities = np.asarray(
        content_similarity[movie_index]
    ).flatten()

    results = movie_features[
        ["movie_id", "title", "genre_text"]
    ].copy()

    results["content_similarity"] = similarities

    selected_movie_id = matches.iloc[0]["movie_id"]

    results = results[
        results["movie_id"] != selected_movie_id
    ]

    return results.sort_values(
        "content_similarity",
        ascending=False
    ).head(top_n).reset_index(drop=True)


# ==========================================
# 4. PERSONALIZED HYBRID RECOMMENDATIONS
# ==========================================

def get_personalized_hybrid_recommendations(
    user_id,
    query,
    top_n=10
):

    user_id = validate_user(user_id)
    query = validate_query(query)

    validate_top_n(top_n)

    collaborative = get_collaborative_recommendations(
        user_id,
        top_n=100
    )

    semantic = get_semantic_scores(query)

    results = collaborative.merge(
        semantic,
        on="movie_id",
        how="inner"
    )

    results = results.merge(
        movie_features[
            [
                "movie_id",
                "rating_score",
                "popularity_score"
            ]
        ],
        on="movie_id",
        how="left"
    )

    results["collaborative_score"] = (
        results["predicted_rating"] / 5
    ).clip(0, 1)

    results["content_score"] = (
        0.70 * results["semantic_similarity"]
        + 0.20 * results["rating_score"]
        + 0.10 * results["popularity_score"]
    )

    results["hybrid_score"] = (
        0.60 * results["collaborative_score"]
        + 0.40 * results["content_score"]
    )

    return results.sort_values(
        "hybrid_score",
        ascending=False
    ).head(top_n).reset_index(drop=True)


# ==========================================
# 5. RETRIEVE MOVIES FOR RAG
# ==========================================

def retrieve_personalized_movies(
    user_id,
    query,
    top_n=5
):

    validate_top_n(top_n)

    recommendations = get_personalized_hybrid_recommendations(
        user_id,
        query,
        top_n
    )

    recommendations = recommendations.merge(
        semantic_features[
            ["movie_id", "average_rating"]
        ],
        on="movie_id",
        how="left"
    )

    return recommendations[
        [
            "movie_id",
            "title",
            "genre_text",
            "average_rating",
            "semantic_similarity",
            "hybrid_score"
        ]
    ]


# ==========================================
# 6. ENGINE INFORMATION
# ==========================================

def get_engine_info():

    return {
        "ratings": len(ratings),
        "users": ratings["user_id"].nunique(),
        "movies": movie_features["movie_id"].nunique(),
        "embeddings": movie_embeddings.shape,
        "svd_predictions": np.asarray(svd_predictions).shape
    }