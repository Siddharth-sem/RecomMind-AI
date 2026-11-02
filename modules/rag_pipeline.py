import re
import pandas as pd

from modules.movie_recommender import retrieve_personalized_movies
from modules.llm_engine import generate_movie_explanation


# ==========================================
# EXPLANATION VALIDATION
# ==========================================

def validate_explanation(explanation, movie_title, genres):

    if not isinstance(explanation, str):
        return False

    text = explanation.strip()

    # Reject empty responses
    if not text:
        return False

    # Reject excessively long responses
    if len(text) > 700:
        return False

    # Reject responses that look like incomplete instructions
    bad_patterns = [
        "as an ai",
        "i cannot",
        "i don't know",
        "here is the explanation",
        "title:",
        "average rating:",
        "user preferences:"
    ]

    lowered = text.lower()

    if any(pattern in lowered for pattern in bad_patterns):
        return False

    # Require the explanation to refer to the recommended movie
    normalized_title = re.sub(
        r"[^a-z0-9 ]",
        "",
        movie_title.lower()
    )

    normalized_text = re.sub(
        r"[^a-z0-9 ]",
        "",
        lowered
    )

    title_words = normalized_title.split()

    if not title_words:
        return False

    # At least two title words must appear, or the complete
    # title when it contains only one word.
    required_words = min(2, len(title_words))

    matching_words = sum(
        word in normalized_text
        for word in title_words
    )

    if matching_words < required_words:
        return False

    # Reject obvious attempts to invent plot/production details.
    unsupported_patterns = [
        r"\bdirected by\b",
        r"\bstarring\b",
        r"\bplayed by\b",
        r"\bthe protagonist\b",
        r"\bthe main character\b",
        r"\bin the movie,\b",
        r"\bin this film,\b",
        r"\bthe story follows\b",
        r"\bthe plot follows\b"
    ]

    if any(
        re.search(pattern, lowered)
        for pattern in unsupported_patterns
    ):
        return False

    return True


# ==========================================
# SAFE FALLBACK EXPLANATION
# ==========================================

def create_fallback_explanation(query, movie):

    title = movie["title"]
    genres = movie["genre_text"]
    rating = float(movie["average_rating"])

    query_words = {
        word.lower().strip(".,!?")
        for word in query.split()
        if len(word) > 3
    }

    genre_words = {
        word.lower()
        for word in genres.split()
    }

    matched_preferences = query_words.intersection(
        genre_words
    )

    if matched_preferences:

        preference_text = ", ".join(
            sorted(matched_preferences)
        )

        explanation = (
            f"{title} is tagged with {genres} genres. "
            f"These overlap with your stated interests in "
            f"{preference_text}. "
        )

    else:

        explanation = (
            f"{title} is tagged with {genres} genres. "
            f"It is included based on your personalized "
            f"recommendation ranking. "
        )

    explanation += (
        f"Its average MovieLens rating is {rating:.2f}/5."
    )

    return explanation


# ==========================================
# GENERATE RAG RECOMMENDATIONS
# ==========================================

def generate_rag_recommendations(user_id, query, top_n=5):

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Please enter your preferences.")

    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n must be a positive integer.")

    query = query.strip()

    # Step 1: Retrieve recommendations from trusted data
    movies = retrieve_personalized_movies(
        user_id=user_id,
        query=query,
        top_n=top_n
    )

    if movies.empty:
        return pd.DataFrame()

    results = []

    # Step 2: Process each movie
    for _, movie in movies.iterrows():

        # Metadata always comes from MovieLens
        trusted_title = str(movie["title"])
        trusted_genres = str(movie["genre_text"])
        trusted_rating = float(movie["average_rating"])

        # Step 3: Generate AI explanation
        try:

            explanation = generate_movie_explanation(
                query=query,
                movie_title=trusted_title,
                genres=trusted_genres,
                rating=trusted_rating
            )

            # Step 4: Validate generated output
            is_valid = validate_explanation(
                explanation,
                trusted_title,
                trusted_genres
            )

            if not is_valid:

                explanation = create_fallback_explanation(
                    query,
                    movie
                )

                explanation_source = "Verified fallback"

            else:

                explanation_source = "Qwen + LoRA"

        except Exception:

            # Generation failure must not destroy recommendations
            explanation = create_fallback_explanation(
                query,
                movie
            )

            explanation_source = "Verified fallback"

        # Step 5: Assemble trusted final result
        results.append({

            "movie_id": int(movie["movie_id"]),

            "title": trusted_title,

            "genres": trusted_genres,

            "rating": round(trusted_rating, 2),

            "semantic_similarity": round(
                float(movie["semantic_similarity"]), 4
            ),

            "hybrid_score": round(
                float(movie["hybrid_score"]), 4
            ),

            "explanation": explanation,

            "explanation_source": explanation_source

        })

    return pd.DataFrame(results)