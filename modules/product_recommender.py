import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

RATINGS_PATH = "datasets/products/processed/electronics_reviews.csv"

ratings = pd.read_csv(RATINGS_PATH)

print("Product collaborative engine loaded!")
print("Interactions:", len(ratings))
print("Users:", ratings["user_id"].nunique())
print("Products:", ratings["parent_asin"].nunique())


# Create user-product matrix
user_product_matrix = ratings.pivot_table(
    index="user_id",
    columns="parent_asin",
    values="rating",
    aggfunc="mean"
).fillna(0)


# Convert to sparse matrix
sparse_matrix = csr_matrix(user_product_matrix.values)


# Train SVD
svd = TruncatedSVD(
    n_components=50,
    random_state=42
)

user_factors = svd.fit_transform(sparse_matrix)
product_factors = svd.components_

predicted_ratings = np.dot(
    user_factors,
    product_factors
)

print("SVD model trained!")
print("Matrix shape:", user_product_matrix.shape)
print("Explained variance:", svd.explained_variance_ratio_.sum())


def collaborative_product_recommendations(user_id, top_n=10):

    if user_id not in user_product_matrix.index:
        raise ValueError("User not found in product dataset.")

    user_index = user_product_matrix.index.get_loc(user_id)

    scores = predicted_ratings[user_index]

    already_rated = user_product_matrix.iloc[user_index].values > 0

    scores = scores.copy()
    scores[already_rated] = -np.inf

    indices = np.argsort(scores)[::-1][:top_n]

    results = pd.DataFrame({
        "parent_asin": user_product_matrix.columns[indices],
        "collaborative_score": scores[indices]
    })

    return results.merge(
        pd.read_csv("datasets/products/processed/electronics_catalog.csv"),
        on="parent_asin",
        how="left"
    )