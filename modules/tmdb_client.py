import os
import requests
import re
import difflib

from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

BASE_URL = "https://api.themoviedb.org/3"
IMAGE_URL = "https://image.tmdb.org/t/p/w500"


def search_movie(title):

    if not TMDB_API_KEY:
        raise ValueError("TMDB API key not found in .env")

    # Extract year from MovieLens title
    year_match = re.search(r"\((\d{4})\)$", title)

    year = year_match.group(1) if year_match else None

    # Remove year from title
    clean_title = re.sub(r"\s*\(\d{4}\)$", "", title).strip()

    search_titles = [clean_title]

    if title != clean_title:
        search_titles.append(title)

    for search_title in search_titles:

        url = f"{BASE_URL}/search/movie"

        params = {
            "api_key": TMDB_API_KEY,
            "query": search_title,
            "include_adult": False,
            "language": "en-US"
        }

        if year:
            params["year"] = year

        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        response.raise_for_status()

        results = response.json().get("results", [])

        if not results:
            continue

        # Match returned titles against requested title
        normalized_title = clean_title.lower()

        exact_matches = [
            movie for movie in results
            if movie.get("title", "").lower() == normalized_title
        ]

        candidates = exact_matches if exact_matches else results

        # Prefer matching release year
        if year:
            year_matches = [
                movie for movie in candidates
                if movie.get("release_date", "").startswith(year)
            ]

            if year_matches:
                candidates = year_matches

        movie = candidates[0]

        return {
            "id": movie.get("id"),
            "title": movie.get("title"),
            "overview": movie.get("overview"),
            "release_date": movie.get("release_date"),
            "rating": movie.get("vote_average"),
            "poster_path": movie.get("poster_path"),
            "poster_url": get_poster_url(movie.get("poster_path")),
            "backdrop_url": get_backdrop_url(movie.get("backdrop_path"))
        }

    return None


def get_poster_url(poster_path):

    if not poster_path:
        return None

    return f"{IMAGE_URL}{poster_path}"


def get_backdrop_url(backdrop_path):

    if not backdrop_path:
        return None

    return f"https://image.tmdb.org/t/p/w1280{backdrop_path}"