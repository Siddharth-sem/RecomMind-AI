import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

from modules.movie_recommender import (
    ratings,
    user_means,
    svd_predictions,
    improved_svd_movie_ids,
    movie_features,
    semantic_features,
    movie_embeddings,
    content_similarity,
    get_embedding_model
)


# ==========================================
# CONFIGURATION
# ==========================================

RELEVANT_RATING = 4.0

COLLABORATIVE_WEIGHT = 0.60
SEMANTIC_WEIGHT = 0.28
RATING_WEIGHT = 0.08
POPULARITY_WEIGHT = 0.04


# ==========================================
# CREATE USER PREFERENCE QUERY
# ==========================================

def create_user_query(user_ratings):

    liked_movies = user_ratings[
        user_ratings["rating"] >= RELEVANT_RATING
    ]

    if liked_movies.empty:
        liked_movies = user_ratings.nlargest(
            10, "rating"
        )

    profile = liked_movies.merge(
        movie_features[["movie_id", "genre_text"]],
        on="movie_id",
        how="left"
    )

    genre_counts = {}

    for genres in profile["genre_text"].dropna():

        for genre in genres.split():

            genre_counts[genre] = (
                genre_counts.get(genre, 0) + 1
            )

    if not genre_counts:
        return None

    preferred_genres = sorted(
        genre_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    genre_text = ", ".join(
        genre for genre, count in preferred_genres
    )

    return (
        f"Recommend movies with genres and themes "
        f"related to: {genre_text}"
    )


# ==========================================
# EVALUATION METRICS
# ==========================================

def calculate_precision(recommended, relevant, k):

    hits = len(set(recommended[:k]) & relevant)

    return hits / k


def calculate_recall(recommended, relevant, k):

    if not relevant:
        return 0.0

    hits = len(set(recommended[:k]) & relevant)

    return hits / len(relevant)


def calculate_ndcg(recommended, relevant, k):

    recommended = recommended[:k]

    dcg = 0.0

    for position, movie_id in enumerate(recommended):

        if movie_id in relevant:

            dcg += 1 / np.log2(position + 2)

    ideal_hits = min(len(relevant), k)

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1 / np.log2(position + 2)
        for position in range(ideal_hits)
    )

    return dcg / idcg


# ==========================================
# RECOMMENDATION DIVERSITY
# ==========================================

def calculate_diversity(recommended):

    if len(recommended) < 2:
        return 0.0

    movie_indices = (
        movie_features.reset_index()
        .set_index("movie_id")["index"]
    )

    indices = [
        movie_indices[movie_id]
        for movie_id in recommended
        if movie_id in movie_indices.index
    ]

    if len(indices) < 2:
        return 0.0

    similarities = np.asarray(
        content_similarity
    )[np.ix_(indices, indices)]

    upper_indices = np.triu_indices(
        len(indices), k=1
    )

    pairwise_similarities = similarities[upper_indices]

    return float(
        np.mean(1 - pairwise_similarities)
    )


# ==========================================
# EVALUATE ONE RECOMMENDATION MODEL
# ==========================================

def evaluate_model(recommendation_function, test_users, k):

    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    diversity_scores = []

    recommended_catalogue = set()

    evaluated_users = 0

    for user_id, relevant_movies in test_users.items():

        recommendations = recommendation_function(
            user_id
        )

        if not recommendations:
            continue

        evaluated_users += 1

        precision_scores.append(
            calculate_precision(
                recommendations,
                relevant_movies,
                k
            )
        )

        recall_scores.append(
            calculate_recall(
                recommendations,
                relevant_movies,
                k
            )
        )

        ndcg_scores.append(
            calculate_ndcg(
                recommendations,
                relevant_movies,
                k
            )
        )

        diversity_scores.append(
            calculate_diversity(
                recommendations
            )
        )

        recommended_catalogue.update(
            recommendations[:k]
        )

    if evaluated_users == 0:
        raise ValueError("No users could be evaluated.")

    return {
        "Precision@10": np.mean(precision_scores),
        "Recall@10": np.mean(recall_scores),
        "NDCG@10": np.mean(ndcg_scores),
        "Diversity": np.mean(diversity_scores),
        "Catalogue Coverage": (
            len(recommended_catalogue)
            / movie_features["movie_id"].nunique()
        ),
        "Users Evaluated": evaluated_users
    }


# ==========================================
# MAIN EVALUATION PIPELINE
# ==========================================

def evaluate_recommendation_models(k=10, max_users=None):

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")

    if max_users is not None and (
        isinstance(max_users, bool)
        or not isinstance(max_users, int)
        or max_users <= 0
    ):
        raise ValueError("max_users must be a positive integer or None.")

    print("\nStarting recommendation evaluation...")

    # --------------------------------------
    # 1. Reproduce original train/test split
    # --------------------------------------

    train_data, test_data = train_test_split(
        ratings,
        test_size=0.20,
        random_state=42
    )

    print("Training ratings:", len(train_data))
    print("Testing ratings:", len(test_data))

    # --------------------------------------
    # 2. Identify relevant held-out movies
    # --------------------------------------

    relevant_data = test_data[
        test_data["rating"] >= RELEVANT_RATING
    ]

    test_users = (
        relevant_data.groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )

    # Only users represented in the saved SVD model
    test_users = {
        user_id: movies
        for user_id, movies in test_users.items()
        if user_id in user_means.index
    }

    if max_users is not None:

        rng = np.random.default_rng(42)

        available_users = list(test_users.keys())

        selected_users = rng.choice(
            available_users,
            size=min(max_users, len(available_users)),
            replace=False
        )

        selected_users = sorted(selected_users.tolist())

        test_users = {
            user_id: test_users[user_id]
            for user_id in selected_users
        }

    print("Users selected:", len(test_users))

    # --------------------------------------
    # 3. Build training-only user profiles
    # --------------------------------------

    queries = {}
    seen_movies = {}

    for user_id, user_ratings in train_data.groupby("user_id"):

        if user_id not in test_users:
            continue

        query = create_user_query(user_ratings)

        if query is None:
            continue

        queries[user_id] = query

        seen_movies[user_id] = set(
            user_ratings["movie_id"]
        )

    # Exclude movies already viewed during training
    test_users = {
        user_id: relevant
        for user_id, relevant in test_users.items()
        if user_id in queries
    }

    queries = {
        user_id: queries[user_id]
        for user_id in test_users
    }

    if not test_users:
        raise ValueError("No eligible users found for evaluation.")

    # --------------------------------------
    # 4. Generate semantic embeddings
    # --------------------------------------

    print("Generating user preference embeddings...")

    embedding_model = get_embedding_model()

    user_ids = list(queries.keys())

    query_embeddings = embedding_model.encode(
        [queries[user_id] for user_id in user_ids],
        convert_to_numpy=True,
        batch_size=32,
        show_progress_bar=True
    )

    semantic_matrix = cosine_similarity(
        query_embeddings,
        movie_embeddings
    )

    # Map movie IDs to semantic matrix positions
    semantic_positions = {
        movie_id: index
        for index, movie_id in enumerate(
            semantic_features["movie_id"]
        )
    }

    feature_positions = {
        movie_id: index
        for index, movie_id in enumerate(
            movie_features["movie_id"]
        )
    }

    # --------------------------------------
    # 5. Prepare model predictions
    # --------------------------------------

    movie_id_positions = {
        movie_id: index
        for index, movie_id in enumerate(
            improved_svd_movie_ids
        )
    }

    feature_lookup = movie_features.set_index("movie_id")

    def get_rankings(user_id):

        user_index = user_means.index.get_loc(user_id)

        predictions = np.asarray(
            svd_predictions[user_index]
        ).reshape(-1)

        prediction_lookup = dict(
            zip(improved_svd_movie_ids, predictions)
        )

        semantic_row = semantic_matrix[
            user_ids.index(user_id)
        ]

        semantic_lookup = {
            movie_id: semantic_row[position]
            for movie_id, position in semantic_positions.items()
        }

        candidates = [
            movie_id
            for movie_id in improved_svd_movie_ids
            if movie_id not in seen_movies[user_id]
        ]

        if not candidates:
            return [], [], []

        collaborative_scores = np.array([
            prediction_lookup[movie_id]
            for movie_id in candidates
        ])

        collaborative_scores = np.clip(
            collaborative_scores / 5.0,
            0,
            1
        )

        semantic_scores = np.array([
            semantic_lookup[movie_id]
            for movie_id in candidates
        ])

        rating_scores = np.array([
            feature_lookup.loc[movie_id, "rating_score"]
            for movie_id in candidates
        ])

        popularity_scores = np.array([
            feature_lookup.loc[movie_id, "popularity_score"]
            for movie_id in candidates
        ])

        content_scores = (
            0.70 * semantic_scores
            + 0.20 * rating_scores
            + 0.10 * popularity_scores
        )

        hybrid_scores = (
            COLLABORATIVE_WEIGHT * collaborative_scores
            + 0.40 * content_scores
        )

        # Rank independently
        collaborative_ranked = [
            candidates[i]
            for i in np.argsort(
                collaborative_scores
            )[::-1][:k]
        ]

        semantic_ranked = [
            movie_id
            for movie_id in sorted(
                candidates,
                key=lambda x: semantic_lookup[x],
                reverse=True
            )[:k]
        ]

        hybrid_ranked = [
            candidates[i]
            for i in np.argsort(
                hybrid_scores
            )[::-1][:k]
        ]

        return (
            collaborative_ranked,
            semantic_ranked,
            hybrid_ranked
        )

    # --------------------------------------
    # 6. Evaluate all three approaches
    # --------------------------------------

    print("Calculating recommendation rankings...")

    rankings = {
        "Collaborative Filtering": {},
        "Semantic Recommendations": {},
        "Hybrid Recommendations": {}
    }

    for user_id in test_users:

        collaborative, semantic, hybrid = get_rankings(
            user_id
        )

        rankings["Collaborative Filtering"][user_id] = collaborative
        rankings["Semantic Recommendations"][user_id] = semantic
        rankings["Hybrid Recommendations"][user_id] = hybrid

    # --------------------------------------
    # 7. Calculate evaluation metrics
    # --------------------------------------

    rows = []

    for model_name, model_rankings in rankings.items():

        result = evaluate_model(
            lambda user_id: model_rankings[user_id],
            test_users,
            k
        )

        result["Model"] = model_name

        rows.append(result)

    results = pd.DataFrame(rows)

    results = results[
        [
            "Model",
            "Precision@10",
            "Recall@10",
            "NDCG@10",
            "Diversity",
            "Catalogue Coverage",
            "Users Evaluated"
        ]
    ]

    return results