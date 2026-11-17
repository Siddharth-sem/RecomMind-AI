import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ratings = pd.read_csv(
    os.path.join(BASE_DIR, "datasets", "movie_ratings.csv")
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
N_COMPONENTS = 50
TOP_N = 10
RELEVANT_THRESHOLD = 4


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

train = pd.concat(train_parts)
test = pd.concat(test_parts)

print("Training ratings:", len(train))
print("Testing ratings:", len(test))


# ==========================================
# BUILD MATRIX
# ==========================================

user_ids = np.sort(ratings.user_id.unique())
movie_ids = np.sort(ratings.movie_id.unique())

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

matrix = np.ascontiguousarray(
    rating_matrix.to_numpy(dtype=np.float64)
)

print("Training matrix:", matrix.shape)


# ==========================================
# TRAIN NMF
# ==========================================

print("\nTraining NMF model...")

nmf_model = NMF(
    n_components=N_COMPONENTS,
    init="nndsvda",
    solver="mu",
    beta_loss="frobenius",
    max_iter=500,
    tol=1e-4,
    random_state=RANDOM_STATE
)

user_factors = nmf_model.fit_transform(
    np.ascontiguousarray(matrix)
)

item_factors = nmf_model.components_

predictions = np.dot(
    user_factors,
    item_factors
)

print("NMF training complete!")
print("Reconstruction error:", nmf_model.reconstruction_err_)


# ==========================================
# TEST DATA
# ==========================================

test_by_user = {
    user_id: group
    for user_id, group in test.groupby("user_id")
}


# ==========================================
# EVALUATION
# ==========================================

precisions = []
recalls = []
ndcgs = []

recommended_items = set()

evaluated_users = 0

for user_id, user_test in test_by_user.items():

    user_index = rating_matrix.index.get_loc(user_id)

    relevant = set(
        user_test.loc[
            user_test.rating >= RELEVANT_THRESHOLD,
            "movie_id"
        ]
    )

    if not relevant:
        continue

    scores = predictions[user_index].copy()

    # Exclude training ratings
    scores[matrix[user_index] > 0] = -np.inf

    recommended_indices = np.argsort(scores)[::-1][:TOP_N]

    recommended = movie_ids[recommended_indices].tolist()

    hits = len(set(recommended) & relevant)

    precision = hits / TOP_N
    recall = hits / len(relevant)

    dcg = sum(
        1 / np.log2(rank + 2)
        for rank, movie_id in enumerate(recommended)
        if movie_id in relevant
    )

    ideal_hits = min(len(relevant), TOP_N)

    idcg = sum(
        1 / np.log2(rank + 2)
        for rank in range(ideal_hits)
    )

    ndcg = dcg / idcg if idcg else 0

    precisions.append(precision)
    recalls.append(recall)
    ndcgs.append(ndcg)

    recommended_items.update(recommended)

    evaluated_users += 1


# ==========================================
# RESULTS
# ==========================================

results = pd.DataFrame([{
    "algorithm": "NMF",
    "users_evaluated": evaluated_users,
    "precision@10": np.mean(precisions),
    "recall@10": np.mean(recalls),
    "ndcg@10": np.mean(ndcgs),
    "coverage": len(recommended_items) / len(movie_ids),
    "reconstruction_error": nmf_model.reconstruction_err_
}])


results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "movie_nmf_evaluation_results.csv"
    ),
    index=False
)

print("\n========== NMF RESULTS ==========")
print(results.to_string(index=False))

print("\nResults saved successfully!")