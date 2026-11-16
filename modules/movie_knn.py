import os
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


# ==========================================
# PATHS AND DATA
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "datasets")

ratings = pd.read_csv(
    os.path.join(DATASET_DIR, "movie_ratings.csv")
)

required_columns = {"user_id", "movie_id", "rating"}

if not required_columns.issubset(ratings.columns):
    raise ValueError("Movie ratings dataset has missing columns.")


# ==========================================
# RATING MATRIX
# ==========================================

rating_matrix = ratings.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating",
    fill_value=0
)

user_item_matrix = csr_matrix(
    rating_matrix.values,
    dtype=np.float32
)

item_user_matrix = user_item_matrix.T.tocsr()

user_ids = rating_matrix.index.to_numpy()
movie_ids = rating_matrix.columns.to_numpy()


# ==========================================
# USER-BASED KNN MODEL
# ==========================================

user_knn = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_jobs=-1
)

user_knn.fit(user_item_matrix)


# ==========================================
# ITEM-BASED KNN MODEL
# ==========================================

item_knn = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_jobs=-1
)

item_knn.fit(item_user_matrix)


# ==========================================
# VALIDATION
# ==========================================

def validate_user(user_id):

    if user_id not in rating_matrix.index:
        raise ValueError(f"User ID {user_id} not found.")

    return rating_matrix.index.get_loc(user_id)


def validate_top_n(top_n):

    if not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n must be a positive integer.")


# ==========================================
# USER-BASED KNN RECOMMENDATIONS
# ==========================================

def user_based_knn_recommendations(
    user_id,
    top_n=10,
    n_neighbors=20
):

    user_index = validate_user(user_id)
    validate_top_n(top_n)

    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive.")

    neighbors_count = min(
        n_neighbors + 1,
        len(user_ids)
    )

    distances, indices = user_knn.kneighbors(
        user_item_matrix[user_index],
        n_neighbors=neighbors_count
    )

    neighbor_indices = indices.flatten()[1:]
    similarities = 1 - distances.flatten()[1:]

    neighbor_ratings = user_item_matrix[neighbor_indices].toarray()

    # Weighted rating predictions
    weighted_scores = np.dot(
        similarities,
        neighbor_ratings
    )

    similarity_sum = np.sum(
        np.abs(similarities)
    )

    if similarity_sum == 0:
        return pd.DataFrame(
            columns=["movie_id", "predicted_rating"]
        )

    predictions = weighted_scores / similarity_sum

    # Exclude movies already rated
    watched = user_item_matrix[user_index].toarray().flatten() > 0

    predictions[watched] = -np.inf

    top_indices = np.argsort(predictions)[::-1][:top_n]

    valid_indices = [
        i for i in top_indices
        if np.isfinite(predictions[i])
    ]

    return pd.DataFrame({
        "movie_id": movie_ids[valid_indices],
        "predicted_rating": predictions[valid_indices]
    }).reset_index(drop=True)


# ==========================================
# ITEM-BASED KNN RECOMMENDATIONS
# ==========================================

def item_based_knn_recommendations(
    user_id,
    top_n=10,
    n_neighbors=20
):

    user_index = validate_user(user_id)
    validate_top_n(top_n)

    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive.")

    user_ratings = user_item_matrix[user_index].toarray().flatten()

    rated_indices = np.where(user_ratings > 0)[0]

    if len(rated_indices) == 0:
        return pd.DataFrame(
            columns=["movie_id", "predicted_rating"]
        )

    scores = np.zeros(len(movie_ids), dtype=np.float32)
    similarity_totals = np.zeros(len(movie_ids), dtype=np.float32)

    # Find similar movies for each movie rated by the user
    for item_index in rated_indices:

        neighbors_count = min(
            n_neighbors + 1,
            len(movie_ids)
        )

        distances, indices = item_knn.kneighbors(
            item_user_matrix[item_index],
            n_neighbors=neighbors_count
        )

        neighbor_indices = indices.flatten()
        similarities = 1 - distances.flatten()

        for neighbor_index, similarity in zip(
            neighbor_indices,
            similarities
        ):

            if neighbor_index == item_index:
                continue

            scores[neighbor_index] += (
                similarity * user_ratings[item_index]
            )

            similarity_totals[neighbor_index] += abs(similarity)

    predictions = np.divide(
        scores,
        similarity_totals,
        out=np.zeros_like(scores),
        where=similarity_totals != 0
    )

    # Exclude already-rated movies
    predictions[user_ratings > 0] = -np.inf

    top_indices = np.argsort(predictions)[::-1][:top_n]

    valid_indices = [
        i for i in top_indices
        if np.isfinite(predictions[i])
    ]

    return pd.DataFrame({
        "movie_id": movie_ids[valid_indices],
        "predicted_rating": predictions[valid_indices]
    }).reset_index(drop=True)


print("Movie KNN models loaded successfully!")
print("Rating matrix:", rating_matrix.shape)