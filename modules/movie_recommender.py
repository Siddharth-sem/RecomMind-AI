import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "models")


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


print("Movie datasets loaded successfully!")
print("Ratings:", ratings.shape)
print("Movie features:", movie_features.shape)
print("Semantic features:", semantic_features.shape)
print("Embeddings:", movie_embeddings.shape)


# ==========================================
# LOAD TRAINED RECOMMENDATION MODELS
# ==========================================

content_similarity = joblib.load(
    os.path.join(MODEL_DIR, "cosine_similarity.pkl")
)

svd_model = joblib.load(
    os.path.join(MODEL_DIR, "improved_svd_model.pkl")
)

svd_predictions = joblib.load(
    os.path.join(MODEL_DIR, "improved_svd_predictions.pkl")
)

user_means = joblib.load(
    os.path.join(MODEL_DIR, "user_means.pkl")
)

user_similarity = joblib.load(
    os.path.join(MODEL_DIR, "user_similarity.pkl")
)

item_similarity = joblib.load(
    os.path.join(MODEL_DIR, "item_similarity.pkl")
)

improved_svd_movie_ids = np.load(
    os.path.join(MODEL_DIR, "improved_svd_movie_ids.npy")
)

tfidf_vectorizer = joblib.load(
    os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
)


print("\nAll recommendation models loaded successfully!")

print("Content similarity:", content_similarity.shape)
print("SVD predictions:", svd_predictions.shape)
print("User similarity:", user_similarity.shape)
print("Item similarity:", item_similarity.shape)
print("Movie ID mapping:", improved_svd_movie_ids.shape)
print("User means:", user_means.shape)



# ==========================================
# LOAD SEMANTIC EMBEDDING MODEL
# ==========================================

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

print("\nSemantic embedding model loaded successfully!")
print("Embedding dimensions:", embedding_model.get_sentence_embedding_dimension())

def get_collaborative_recommendations(user_id, top_n=10):

    if user_id not in user_means.index:
        raise ValueError(f"User ID {user_id} not found.")

    user_index = user_means.index.get_loc(user_id)

    predictions = svd_predictions[user_index]

    recommendations = pd.DataFrame({
        "movie_id": improved_svd_movie_ids,
        "predicted_rating": predictions
    })

    # Remove movies already rated by this user
    watched_movies = ratings.loc[
        ratings["user_id"] == user_id, "movie_id"
    ].values

    recommendations = recommendations[
        ~recommendations["movie_id"].isin(watched_movies)
    ]

    # Sort by predicted rating
    recommendations = recommendations.sort_values(
        by="predicted_rating",
        ascending=False
    )

    # Add movie details
    recommendations = recommendations.merge(
        movie_features[["movie_id", "title", "genre_text"]],
        on="movie_id",
        how="left"
    )

    return recommendations.head(top_n).reset_index(drop=True)


def get_semantic_scores(query):

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Please enter a valid search query.")

    # Convert query into embedding
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    # Calculate similarity with all movie embeddings
    similarities = cosine_similarity(
        query_embedding,
        movie_embeddings
    )[0]

    # Create results
    semantic_scores = semantic_features[["movie_id"]].copy()

    semantic_scores["semantic_similarity"] = similarities

    return semantic_scores.sort_values(
        by="semantic_similarity",
        ascending=False
    ).reset_index(drop=True)

def get_personalized_hybrid_recommendations(user_id, query, top_n=10):

    # Get collaborative recommendations
    collaborative = get_collaborative_recommendations(
        user_id,
        top_n=100
    )

    # Get semantic scores
    semantic = get_semantic_scores(query)

    # Merge semantic scores
    results = collaborative.merge(
        semantic,
        on="movie_id",
        how="inner"
    )

    # Add rating and popularity information
    results = results.merge(
        movie_features[
            ["movie_id", "rating_score", "popularity_score"]
        ],
        on="movie_id",
        how="left"
    )

    # Normalize collaborative predictions
    results["collaborative_score"] = (
        results["predicted_rating"] / 5
    ).clip(0, 1)

    # Calculate content score
    results["content_score"] = (
        0.70 * results["semantic_similarity"]
        + 0.20 * results["rating_score"]
        + 0.10 * results["popularity_score"]
    )

    # Final hybrid score
    results["hybrid_score"] = (
        0.60 * results["collaborative_score"]
        + 0.40 * results["content_score"]
    )

    return results.sort_values(
        by="hybrid_score",
        ascending=False
    ).head(top_n).reset_index(drop=True)

def retrieve_personalized_movies(user_id, query, top_n=5):

    recommendations = get_personalized_hybrid_recommendations(
        user_id,
        query,
        top_n
    )

    # Add average ratings
    recommendations = recommendations.merge(
        semantic_features[["movie_id", "average_rating"]],
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