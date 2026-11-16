import os
import numpy as np
import pandas as pd

from sklearn.decomposition import NMF


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    BASE_DIR, "datasets", "movie_ratings.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

ratings = pd.read_csv(DATASET_PATH)

rating_matrix = ratings.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating",
    fill_value=0
)

# Train NMF
nmf_model = NMF(
    n_components=50,
    init="nndsvda",
    solver="cd",
    max_iter=300,
    random_state=42
)

nmf_input = np.ascontiguousarray(
    rating_matrix.values,
    dtype=np.float64
)

user_factors = nmf_model.fit_transform(nmf_input)

item_factors = nmf_model.components_

predicted_ratings = np.ascontiguousarray(
    np.dot(user_factors, item_factors)
)

# Save model components
np.save(
    os.path.join(MODEL_DIR, "movie_nmf_user_factors.npy"),
    user_factors
)

np.save(
    os.path.join(MODEL_DIR, "movie_nmf_item_factors.npy"),
    item_factors
)

np.save(
    os.path.join(MODEL_DIR, "movie_nmf_predictions.npy"),
    predicted_ratings
)

np.save(
    os.path.join(MODEL_DIR, "movie_nmf_user_ids.npy"),
    rating_matrix.index.to_numpy()
)

np.save(
    os.path.join(MODEL_DIR, "movie_nmf_movie_ids.npy"),
    rating_matrix.columns.to_numpy()
)


# ==========================================
# RECOMMENDATIONS
# ==========================================

def nmf_recommendations(user_id, top_n=10):

    if user_id not in rating_matrix.index:
        raise ValueError("User ID not found.")

    if not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n must be positive.")

    user_index = rating_matrix.index.get_loc(user_id)

    scores = predicted_ratings[user_index].copy()

    # Exclude previously rated movies
    watched = rating_matrix.iloc[user_index].values > 0

    scores[watched] = -np.inf

    indices = np.argsort(scores)[::-1][:top_n]

    results = pd.DataFrame({
        "movie_id": rating_matrix.columns[indices],
        "predicted_rating": scores[indices]
    })

    return results.reset_index(drop=True)


print("Movie NMF model trained successfully!")
print("Rating matrix:", rating_matrix.shape)
print("User factors:", user_factors.shape)
print("Item factors:", item_factors.shape)
print("Reconstruction error:", nmf_model.reconstruction_err_)