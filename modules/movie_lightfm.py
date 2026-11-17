import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz


# =========================================================
# PATHS
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = (
    ROOT_DIR
    / "models"
    / "lightfm"
    / "RecomMind_LightFM"
)

PARAMETERS_PATH = MODEL_DIR / "lightfm_parameters.npz"
FEATURES_PATH = MODEL_DIR / "item_features.npz"
MAPPINGS_PATH = MODEL_DIR / "mappings.json"
CATALOG_PATH = MODEL_DIR / "movie_catalog.csv"

RATINGS_PATH = ROOT_DIR / "datasets" / "movie_ratings.csv"


# =========================================================
# MODEL LOADING
# =========================================================

class MovieLightFM:

    def __init__(self):

        self.parameters = None
        self.item_features = None
        self.mappings = None
        self.catalog = None
        self.ratings = None

        self.user_embeddings = None
        self.user_biases = None
        self.item_embeddings = None
        self.item_biases = None

        self.movie_feature_embeddings = None
        self.movie_feature_biases = None

        self._load_model()

    def _load_model(self):

        required_files = [
            PARAMETERS_PATH,
            FEATURES_PATH,
            MAPPINGS_PATH,
            CATALOG_PATH
        ]

        for path in required_files:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required LightFM file missing: {path}"
                )

        # Load parameters
        self.parameters = np.load(
            PARAMETERS_PATH,
            allow_pickle=False
        )

        self.user_embeddings = self.parameters[
            "user_embeddings"
        ]

        self.user_biases = self.parameters[
            "user_biases"
        ]

        self.item_embeddings = self.parameters[
            "item_embeddings"
        ]

        self.item_biases = self.parameters[
            "item_biases"
        ]

        # Load sparse item features
        self.item_features = load_npz(
            FEATURES_PATH
        ).tocsr()

        # Load mappings
        with open(MAPPINGS_PATH, "r", encoding="utf-8") as f:
            self.mappings = json.load(f)

        self.user_id_map = self.mappings["user_id_map"]
        self.item_id_map = self.mappings["item_id_map"]

        # Load movie catalog
        self.catalog = pd.read_csv(CATALOG_PATH)

        self.catalog["movie_id"] = (
            self.catalog["movie_id"].astype(int)
        )

        # Load local ratings dataset for movie titles
        if RATINGS_PATH.exists():

            self.ratings = pd.read_csv(RATINGS_PATH)

            self.ratings["user_id"] = (
                self.ratings["user_id"].astype(int)
            )

            self.ratings["movie_id"] = (
                self.ratings["movie_id"].astype(int)
            )

            # Attach movie titles if catalog doesn't contain them
            if "title" not in self.catalog.columns:

                movie_titles = (
                    self.ratings[["movie_id", "title"]]
                    .drop_duplicates("movie_id")
                )

                self.catalog = self.catalog.merge(
                    movie_titles,
                    on="movie_id",
                    how="left"
                )

        # Verify dimensions
        if self.item_features.shape[0] != len(self.catalog):
            raise ValueError(
                "Item feature rows do not match movie catalog."
            )

        if self.item_features.shape[1] != len(self.item_embeddings):
            raise ValueError(
                "Item feature columns do not match embeddings."
            )

        if self.item_features.shape[1] != len(self.item_biases):
            raise ValueError(
                "Item feature columns do not match biases."
            )

        if self.user_embeddings.shape[0] != len(self.user_biases):
            raise ValueError(
                "User embedding and bias dimensions mismatch."
            )

        # Precompute movie representations
        self.movie_feature_embeddings = (
            self.item_features @ self.item_embeddings
        )

        self.movie_feature_biases = (
            self.item_features @ self.item_biases
        )

        # Load ratings for excluding previously rated movies
        if RATINGS_PATH.exists():

            self.ratings = pd.read_csv(
                RATINGS_PATH,
                usecols=["user_id", "movie_id"]
            )

            self.ratings["user_id"] = (
                self.ratings["user_id"].astype(int)
            )

            self.ratings["movie_id"] = (
                self.ratings["movie_id"].astype(int)
            )

        print("LightFM movie model loaded successfully!")

    # =====================================================
    # RECOMMENDATIONS
    # =====================================================

    def recommend(self, user_id, top_n=10):

        user_key = str(int(user_id))

        if user_key not in self.user_id_map:
            return pd.DataFrame()

        user_index = self.user_id_map[user_key]

        # User representation
        user_embedding = self.user_embeddings[user_index]
        user_bias = self.user_biases[user_index]

        # LightFM prediction formula
        scores = (
            self.movie_feature_embeddings @ user_embedding
            + self.movie_feature_biases
            + user_bias
        )

        scores = np.asarray(scores).ravel()

        # Exclude movies already rated by this user
        seen_movies = set()

        if self.ratings is not None:

            seen_movies = set(
                self.ratings.loc[
                    self.ratings["user_id"] == int(user_id),
                    "movie_id"
                ].tolist()
            )

        # Candidate movies
        candidates = self.catalog.copy()

        candidates["score"] = scores

        candidates = candidates[
            ~candidates["movie_id"].isin(seen_movies)
        ]

        candidates = candidates.sort_values(
            "score",
            ascending=False
        )

        result = candidates.head(top_n).copy()

        return result.reset_index(drop=True)

    # =====================================================
    # SIMILARITY / INFO
    # =====================================================

    def get_model_info(self):

        return {
            "algorithm": "LightFM",
            "loss": "WARP",
            "components": self.user_embeddings.shape[1],
            "users": self.user_embeddings.shape[0],
            "movies": len(self.catalog),
            "features": self.item_features.shape[1]
        }


# =========================================================
# CONVENIENCE FUNCTION
# =========================================================

_lightfm_instance = None


def load_lightfm():

    global _lightfm_instance

    if _lightfm_instance is None:
        _lightfm_instance = MovieLightFM()

    return _lightfm_instance


def get_lightfm_recommendations(user_id, top_n=10):

    model = load_lightfm()

    return model.recommend(
        user_id=user_id,
        top_n=top_n
    )