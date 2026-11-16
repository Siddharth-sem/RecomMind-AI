import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


# ==========================================
# CONFIGURATION
# ==========================================

DATA_DIR = "datasets/books"
MODEL_DIR = "models"

RELEVANT_RATING = 7
RANDOM_STATE = 42
N_COMPONENTS = 50


# ==========================================
# LOAD DATA
# ==========================================

ratings = pd.read_csv(
    f"{DATA_DIR}/clean_ratings.csv",
    dtype={"ISBN": str}
)

features = pd.read_csv(
    f"{MODEL_DIR}/semantic_book_features.csv",
    dtype={"ISBN": str}
)

embeddings = np.load(
    f"{MODEL_DIR}/book_embeddings.npy"
)

# Normalize embeddings for cosine similarity
embeddings = normalize(embeddings)

# Map ISBNs to embedding positions
isbn_to_embedding = {
    isbn: index
    for index, isbn in enumerate(features["ISBN"])
}

# Catalogue of books available for recommendation
all_books = sorted(
    set(ratings["ISBN"]) & set(isbn_to_embedding)
)

isbn_to_index = {
    isbn: index
    for index, isbn in enumerate(all_books)
}

index_to_isbn = np.array(all_books)

print("Evaluation module loaded!")
print("Ratings:", len(ratings))
print("Books:", len(all_books))
print("Embedding shape:", embeddings.shape)


# ==========================================
# EVALUATION METRICS
# ==========================================

def precision_at_k(recommended, relevant, k):

    hits = len(set(recommended[:k]) & relevant)

    return hits / k


def recall_at_k(recommended, relevant, k):

    if not relevant:
        return 0.0

    hits = len(set(recommended[:k]) & relevant)

    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k):

    recommended = recommended[:k]

    dcg = sum(
        1 / np.log2(position + 2)
        for position, isbn in enumerate(recommended)
        if isbn in relevant
    )

    ideal_hits = min(len(relevant), k)

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1 / np.log2(position + 2)
        for position in range(ideal_hits)
    )

    return dcg / idcg


# ==========================================
# MAIN EVALUATION
# ==========================================

def evaluate_book_models(
    k=10,
    max_users=1000
):

    print("\nStarting book recommendation evaluation...")

    # --------------------------------------
    # 1. Train/test split
    # --------------------------------------

    train_data, test_data = train_test_split(
        ratings,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    print("Training ratings:", len(train_data))
    print("Testing ratings:", len(test_data))

    # --------------------------------------
    # 2. Build user and book mappings
    # --------------------------------------

    user_ids = sorted(train_data["User-ID"].unique())

    user_to_index = {
        user_id: index
        for index, user_id in enumerate(user_ids)
    }

    # --------------------------------------
    # 3. Build sparse training matrix
    # --------------------------------------

    rows = train_data["User-ID"].map(user_to_index).to_numpy()

    cols = train_data["ISBN"].map(isbn_to_index).to_numpy()

    values = train_data["Book-Rating"].to_numpy(dtype=float)

    rating_matrix = csr_matrix(
        (values, (rows, cols)),
        shape=(len(user_ids), len(all_books))
    )

    # --------------------------------------
    # 4. Train collaborative model
    # --------------------------------------

    print("\nTraining collaborative filtering model...")

    components = min(
        N_COMPONENTS,
        min(rating_matrix.shape) - 1
    )

    svd = TruncatedSVD(
        n_components=components,
        random_state=RANDOM_STATE
    )

    user_factors = svd.fit_transform(rating_matrix)

    item_factors = svd.components_

    print("Collaborative model trained.")

    # --------------------------------------
    # 5. Identify relevant test books
    # --------------------------------------

    relevant_test = test_data[
        test_data["Book-Rating"] >= RELEVANT_RATING
    ]

    test_users = (
        relevant_test.groupby("User-ID")["ISBN"]
        .apply(set)
        .to_dict()
    )

    # Keep books present in our catalogue
    test_users = {
        user: {
            isbn for isbn in books
            if isbn in isbn_to_index
        }
        for user, books in test_users.items()
    }

    # Remove users not present in training
    test_users = {
        user: books
        for user, books in test_users.items()
        if user in user_to_index and books
    }

    # Select a reproducible sample
    rng = np.random.default_rng(RANDOM_STATE)

    selected_users = sorted(test_users)

    if max_users is not None:

        selected_users = rng.choice(
            selected_users,
            size=min(max_users, len(selected_users)),
            replace=False
        ).tolist()

    print("Users selected:", len(selected_users))

    # --------------------------------------
    # 6. Generate recommendations
    # --------------------------------------

    results = {
        "Collaborative Filtering": [],
        "Semantic Recommendations": [],
        "Hybrid Recommendations": []
    }

    print("\nGenerating recommendations...")

    for count, user_id in enumerate(selected_users, start=1):

        user_index = user_to_index[user_id]

        user_train = train_data[
            train_data["User-ID"] == user_id
        ]

        seen_books = set(user_train["ISBN"])

        relevant_books = test_users[user_id]

        # Candidate books exclude training books
        candidates = [
            isbn for isbn in all_books
            if isbn not in seen_books
        ]

        if not candidates:
            continue

        candidate_indices = np.array([
            isbn_to_index[isbn]
            for isbn in candidates
        ])

        # ----------------------------------
        # Collaborative scores
        # ----------------------------------

        collaborative_scores = (
            user_factors[user_index] @ item_factors
        )[candidate_indices]

        # ----------------------------------
        # Semantic user profile
        # ----------------------------------

        liked_books = user_train[
            user_train["Book-Rating"] >= RELEVANT_RATING
        ]

        # Fallback if user has no highly rated books
        if liked_books.empty:

            liked_books = user_train.nlargest(
                5,
                "Book-Rating"
            )

        liked_indices = [
            isbn_to_embedding[isbn]
            for isbn in liked_books["ISBN"]
            if isbn in isbn_to_embedding
        ]

        if not liked_indices:
            continue

        user_profile = embeddings[liked_indices].mean(axis=0)

        user_profile = normalize(
            user_profile.reshape(1, -1)
        )[0]

        semantic_scores = (
            embeddings[
                [
                    isbn_to_embedding[isbn]
                    for isbn in candidates
                ]
            ] @ user_profile
        )

        # ----------------------------------
        # Normalize scores
        # ----------------------------------

        def minmax(scores):

            minimum = scores.min()
            maximum = scores.max()

            if maximum == minimum:
                return np.zeros_like(scores)

            return (scores - minimum) / (
                maximum - minimum
            )

        cf_normalized = minmax(
            collaborative_scores
        )

        semantic_normalized = minmax(
            semantic_scores
        )

        # ----------------------------------
        # Hybrid score
        # ----------------------------------

        hybrid_scores = (
            0.60 * cf_normalized
            + 0.40 * semantic_normalized
        )

        # ----------------------------------
        # Rank books
        # ----------------------------------

        rankings = {
            "Collaborative Filtering": collaborative_scores,
            "Semantic Recommendations": semantic_scores,
            "Hybrid Recommendations": hybrid_scores
        }

        for model_name, scores in rankings.items():

            top_indices = np.argsort(scores)[::-1][:k]

            recommended = [
                candidates[index]
                for index in top_indices
            ]

            results[model_name].append({
                "precision": precision_at_k(
                    recommended,
                    relevant_books,
                    k
                ),
                "recall": recall_at_k(
                    recommended,
                    relevant_books,
                    k
                ),
                "ndcg": ndcg_at_k(
                    recommended,
                    relevant_books,
                    k
                )
            })

        if count % 50 == 0:
            print(f"Evaluated {count} users...")

    # --------------------------------------
    # 7. Calculate final metrics
    # --------------------------------------

    rows = []

    for model_name, scores in results.items():

        if not scores:
            continue

        rows.append({
            "Model": model_name,
            "Precision@10": np.mean([
                x["precision"] for x in scores
            ]),
            "Recall@10": np.mean([
                x["recall"] for x in scores
            ]),
            "NDCG@10": np.mean([
                x["ndcg"] for x in scores
            ]),
            "Users Evaluated": len(scores)
        })

    evaluation_results = pd.DataFrame(rows)

    return evaluation_results


# ==========================================
# RUN EVALUATION
# ==========================================

if __name__ == "__main__":

    results = evaluate_book_models(
        k=10,
        max_users=1000
    )

    print("\n========== BOOK MODEL EVALUATION ==========")

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    results.to_csv(
        "datasets/books/book_evaluation_results.csv",
        index=False
    )

    print("\nResults saved successfully!")