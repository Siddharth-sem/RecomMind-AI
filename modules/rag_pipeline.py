import pandas as pd

from modules.movie_recommender import retrieve_personalized_movies
from modules.llm_engine import generate_movie_explanation


def generate_rag_recommendations(user_id, query, top_n=5):

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Please enter your preferences.")

    # Step 1: Retrieve relevant movies
    movies = retrieve_personalized_movies(
        user_id=user_id,
        query=query,
        top_n=top_n
    )

    if movies.empty:
        return pd.DataFrame()

    results = []

    # Step 2: Generate an explanation for each movie
    for _, movie in movies.iterrows():

        explanation = generate_movie_explanation(
            query=query,
            movie_title=movie["title"],
            genres=movie["genre_text"],
            rating=float(movie["average_rating"])
        )

        # Step 3: Keep factual metadata from the dataset
        results.append({
            "movie_id": int(movie["movie_id"]),
            "title": movie["title"],
            "genres": movie["genre_text"],
            "rating": round(float(movie["average_rating"]), 2),
            "semantic_similarity": round(
                float(movie["semantic_similarity"]), 4
            ),
            "hybrid_score": round(
                float(movie["hybrid_score"]), 4
            ),
            "explanation": explanation
        })

    return pd.DataFrame(results)