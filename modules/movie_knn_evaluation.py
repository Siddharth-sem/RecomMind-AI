import os
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


# ==========================================
# PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    BASE_DIR, "datasets", "movie_ratings.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "datasets", "evaluations"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# CONFIGURATION
# ==========================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_NEIGHBORS = 20
TOP_N = 10
RELEVANT_THRESHOLD = 4


# ==========================================
# LOAD DATA
# ==========================================

ratings = pd.read_csv(DATASET_PATH)

required_columns = {"user_id", "movie_id", "rating"}

if not required_columns.issubset(ratings.columns):
    raise ValueError("Dataset is missing required columns.")

ratings = ratings.dropna(
    subset=["user_id", "movie_id", "rating"]
).copy()

ratings["user_id"] = ratings["user_id"].astype(int)
ratings["movie_id"] = ratings["movie_id"].astype(int)

print("Dataset loaded:", ratings.shape)


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

train_parts = []
test_parts = []

for user_id, group in ratings.groupby("user_id"):

    if len(group) < 2:
        train_parts.append(group)
        continue

    train_user, test_user = train_test_split(
        group,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_parts.append(train_user)
    test_parts.append(test_user)

train = pd.concat(train_parts).reset_index(drop=True)
test = pd.concat(test_parts).reset_index(drop=True)

print("Training ratings:", len(train))
print("Testing ratings:", len(test))


# ==========================================
# BUILD TRAINING MATRIX
# ==========================================

user_ids = np.sort(ratings["user_id"].unique())
movie_ids = np.sort(ratings["movie_id"].unique())

rating_matrix = (
    train.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating",
        fill_value=0
    )
    .reindex(
        index=user_ids,
        columns=movie_ids,
        fill_value=0
    )
)

matrix = rating_matrix.to_numpy(dtype=np.float32)

print("Training matrix:", matrix.shape)


# ==========================================
# TEST DATA BY USER
# ==========================================

test_by_user = {
    user_id: group
    for user_id, group in test.groupby("user_id")
}


# ==========================================
# USER SIMILARITY
# ==========================================

print("\nCalculating user similarities...")

user_similarity = cosine_similarity(matrix)

np.fill_diagonal(user_similarity, 0)


# ==========================================
# ITEM SIMILARITY
# ==========================================

print("Calculating item similarities...")

item_similarity = cosine_similarity(matrix.T)

np.fill_diagonal(item_similarity, 0)


# ==========================================
# USER-BASED KNN
# ==========================================

def user_based_recommendations(user_index):

    similarities = user_similarity[user_index]

    neighbor_indices = np.argsort(
        similarities
    )[::-1][:N_NEIGHBORS]

    neighbor_scores = similarities[neighbor_indices]

    neighbor_ratings = matrix[neighbor_indices]

    scores = np.dot(
        neighbor_scores,
        neighbor_ratings
    )

    denominator = np.sum(np.abs(neighbor_scores))

    if denominator > 0:
        scores = scores / denominator

    # Exclude movies already rated
    scores[matrix[user_index] > 0] = -np.inf

    # Exclude candidates with no positive score
    scores[~np.isfinite(scores)] = -np.inf

    return np.argsort(scores)[::-1][:TOP_N]

def item_based_recommendations(user_index):

    user_ratings = matrix[user_index]

    rated_indices = np.where(user_ratings > 0)[0]

    if len(rated_indices) == 0:
        return np.array([], dtype=int)

    scores = np.zeros(len(movie_ids), dtype=np.float64)
    similarity_totals = np.zeros(len(movie_ids), dtype=np.float64)

    for item_index in rated_indices:

        similarities = item_similarity[item_index].copy()

        # Select only the strongest similar items
        similarities[item_index] = 0

        neighbor_indices = np.argsort(
            similarities
        )[::-1][:N_NEIGHBORS]

        neighbor_scores = similarities[neighbor_indices]

        # Ignore zero-similarity neighbours
        valid = neighbor_scores > 0

        neighbor_indices = neighbor_indices[valid]
        neighbor_scores = neighbor_scores[valid]

        if len(neighbor_indices) == 0:
            continue

        scores[neighbor_indices] += (
            neighbor_scores * user_ratings[item_index]
        )

        similarity_totals[neighbor_indices] += neighbor_scores

    predictions = np.divide(
        scores,
        similarity_totals,
        out=np.zeros_like(scores),
        where=similarity_totals > 0
    )

    # Exclude previously rated movies
    predictions[user_ratings > 0] = -np.inf

    # Exclude movies without a positive similarity score
    predictions[similarity_totals == 0] = -np.inf

    valid_indices = np.where(np.isfinite(predictions))[0]

    top_indices = valid_indices[
        np.argsort(predictions[valid_indices])[::-1][:TOP_N]
    ]

    return top_indices


# ==========================================
# EVALUATION METRICS
# ==========================================

def precision_at_k(recommended, relevant):

    if len(recommended) == 0:
        return 0.0

    hits = len(set(recommended) & relevant)

    return hits / TOP_N


def recall_at_k(recommended, relevant):

    if not relevant:
        return 0.0

    hits = len(set(recommended) & relevant)

    return hits / len(relevant)


def ndcg_at_k(recommended, relevant):

    dcg = 0.0

    for rank, movie_index in enumerate(recommended):

        if movie_index in relevant:
            dcg += 1 / np.log2(rank + 2)

    ideal_hits = min(len(relevant), TOP_N)

    if ideal_hits == 0:
        return 0.0

    ideal_dcg = sum(
        1 / np.log2(rank + 2)
        for rank in range(ideal_hits)
    )

    return dcg / ideal_dcg


# ==========================================
# RUN EVALUATION
# ==========================================

def evaluate_model(model_name, recommendation_function):

    precisions = []
    recalls = []
    ndcgs = []

    recommended_items = set()

    evaluated_users = 0

    for user_id, user_test in test_by_user.items():

        user_index = rating_matrix.index.get_loc(user_id)

        relevant_movie_ids = set(
            user_test.loc[
                user_test["rating"] >= RELEVANT_THRESHOLD,
                "movie_id"
            ]
        )

        # Users without relevant test items cannot contribute
        # meaningfully to these ranking metrics.
        if not relevant_movie_ids:
            continue

        recommendations = recommendation_function(
            user_index
        )

        recommended_movie_ids = [
            int(movie_ids[index])
            for index in recommendations
        ]

        relevant = relevant_movie_ids

        precisions.append(
            precision_at_k(
                recommended_movie_ids,
                relevant
            )
        )

        recalls.append(
            recall_at_k(
                recommended_movie_ids,
                relevant
            )
        )

        ndcgs.append(
            ndcg_at_k(
                recommended_movie_ids,
                relevant
            )
        )

        recommended_items.update(
            recommended_movie_ids
        )

        evaluated_users += 1

    return {
        "algorithm": model_name,
        "users_evaluated": evaluated_users,
        "precision@10": np.mean(precisions),
        "recall@10": np.mean(recalls),
        "ndcg@10": np.mean(ndcgs),
        "coverage": len(recommended_items) / len(movie_ids)
    }


# ==========================================
# EXECUTION
# ==========================================

print("\nEvaluating User-based KNN...")

user_results = evaluate_model(
    "User-based KNN",
    user_based_recommendations
)

print("Evaluating Item-based KNN...")

item_results = evaluate_model(
    "Item-based KNN",
    item_based_recommendations
)


# ==========================================
# SAVE RESULTS
# ==========================================

results = pd.DataFrame([
    user_results,
    item_results
])

results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "movie_knn_evaluation_results.csv"
    ),
    index=False
)


# ==========================================
# DISPLAY
# ==========================================

print("\n========== MOVIE KNN RESULTS ==========")

print(
    results.to_string(index=False)
)

print("\nResults saved successfully!")