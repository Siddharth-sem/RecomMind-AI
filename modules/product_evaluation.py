import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

RATINGS_PATH = "datasets/products/processed/electronics_reviews.csv"
FEATURES_PATH = "models/semantic_product_features.csv"
EMBEDDINGS_PATH = "models/product_embeddings.npy"

TOP_N = 10
MAX_USERS = 1000

print("Loading product evaluation data...")

ratings = pd.read_csv(RATINGS_PATH)
features = pd.read_csv(FEATURES_PATH)
embeddings = np.load(EMBEDDINGS_PATH)

# Sample users for faster evaluation
user_counts = ratings["user_id"].value_counts()
eligible_users = user_counts[user_counts >= 10].index

rng = np.random.RandomState(42)
selected_users = rng.choice(
    eligible_users,
    size=min(MAX_USERS, len(eligible_users)),
    replace=False
)

ratings = ratings[ratings["user_id"].isin(selected_users)].copy()

# Train/test split per user
train_parts = []
test_parts = []

for user_id, group in ratings.groupby("user_id"):
    if len(group) < 5:
        continue

    train, test = train_test_split(
        group,
        test_size=0.2,
        random_state=42
    )

    train_parts.append(train)
    test_parts.append(test)

train_df = pd.concat(train_parts)
test_df = pd.concat(test_parts)

# Build training user-product matrix
matrix = train_df.pivot_table(
    index="user_id",
    columns="parent_asin",
    values="rating",
    aggfunc="mean"
).fillna(0)

# Train collaborative model only on training data
sparse_matrix = matrix.values

svd = TruncatedSVD(
    n_components=50,
    random_state=42
)

user_factors = svd.fit_transform(sparse_matrix)
predicted_ratings = user_factors @ svd.components_

# Map product IDs to feature rows
product_to_index = {
    asin: idx for idx, asin in enumerate(features["parent_asin"])
}

# Map matrix product IDs to SVD columns
product_columns = list(matrix.columns)

# Semantic user profiles from liked training products
liked_products = train_df[train_df["rating"] >= 4]

# Evaluation metrics
def calculate_metrics(recommended, relevant, k=10):

    recommended = recommended[:k]

    hits = len(set(recommended) & set(relevant))

    precision = hits / k
    recall = hits / len(relevant) if relevant else 0

    dcg = sum(
        1 / np.log2(rank + 2)
        for rank, item in enumerate(recommended)
        if item in relevant
    )

    ideal_hits = min(len(relevant), k)

    idcg = sum(
        1 / np.log2(rank + 2)
        for rank in range(ideal_hits)
    )

    ndcg = dcg / idcg if idcg else 0

    return precision, recall, ndcg


results = {
    "Collaborative": [],
    "Semantic": [],
    "Hybrid": []
}

print("Evaluating recommendation models...")

for user_id in matrix.index:

    user_index = matrix.index.get_loc(user_id)

    train_seen = set(
        matrix.columns[matrix.loc[user_id] > 0]
    )

    test_user = test_df[test_df["user_id"] == user_id]

    relevant = set(
        test_user.loc[
            test_user["rating"] >= 4,
            "parent_asin"
        ]
    )

    if not relevant:
        continue

    # Candidate products: items not seen during training
    candidates = [
        asin for asin in product_columns
        if asin not in train_seen
    ]

    if not candidates:
        continue

    # Collaborative scores
    cf_scores = predicted_ratings[user_index]

    cf_ranking = sorted(
        candidates,
        key=lambda x: cf_scores[product_columns.index(x)],
        reverse=True
    )[:TOP_N]

    # Semantic profile from liked training products
    user_liked = liked_products[
        liked_products["user_id"] == user_id
    ]["parent_asin"].tolist()

    liked_indices = [
        product_to_index[x]
        for x in user_liked
        if x in product_to_index
    ]

    candidate_indices = [
        product_to_index[x]
        for x in candidates
        if x in product_to_index
    ]

    semantic_ranking = []
    semantic_scores = {}

    if liked_indices and candidate_indices:

        profile = embeddings[liked_indices].mean(axis=0)
        profile = profile / (np.linalg.norm(profile) + 1e-10)

        scores = cosine_similarity(
            profile.reshape(1, -1),
            embeddings[candidate_indices]
        ).flatten()

        semantic_scores = dict(
            zip(
                [features.iloc[i]["parent_asin"] for i in candidate_indices],
                scores
            )
        )

        semantic_ranking = sorted(
            semantic_scores,
            key=semantic_scores.get,
            reverse=True
        )[:TOP_N]

    # Hybrid: normalized CF + semantic ranking scores
    cf_candidate_scores = {
        asin: cf_scores[product_columns.index(asin)]
        for asin in candidates
    }

    cf_min = min(cf_candidate_scores.values())
    cf_max = max(cf_candidate_scores.values())

    cf_normalized = {
        asin: (score - cf_min) / (cf_max - cf_min + 1e-10)
        for asin, score in cf_candidate_scores.items()
    }

    sem_values = list(semantic_scores.values())

    if sem_values:
        sem_min = min(sem_values)
        sem_max = max(sem_values)

        sem_normalized = {
            asin: (score - sem_min) / (sem_max - sem_min + 1e-10)
            for asin, score in semantic_scores.items()
        }
    else:
        sem_normalized = {}

    hybrid_ranking = sorted(
        candidates,
        key=lambda asin: (
            0.4 * cf_normalized.get(asin, 0)
            + 0.6 * sem_normalized.get(asin, 0)
        ),
        reverse=True
    )[:TOP_N]

    # Store metrics
    for model_name, ranking in [
        ("Collaborative", cf_ranking),
        ("Semantic", semantic_ranking),
        ("Hybrid", hybrid_ranking)
    ]:

        metrics = calculate_metrics(
            ranking,
            relevant,
            TOP_N
        )

        results[model_name].append(metrics)


# Summarize
summary = []

for model_name, values in results.items():

    if not values:
        continue

    values = np.array(values)

    summary.append({
        "Model": model_name,
        "Precision@10": values[:, 0].mean(),
        "Recall@10": values[:, 1].mean(),
        "NDCG@10": values[:, 2].mean(),
        "Users Evaluated": len(values)
    })

summary_df = pd.DataFrame(summary)

print("\nPRODUCT RECOMMENDATION EVALUATION")
print(summary_df.to_string(index=False))

summary_df.to_csv(
    "datasets/products/product_evaluation_results.csv",
    index=False
)

print("\nResults saved successfully!")