import os
import requests

from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

BASE_URL = "https://api.themoviedb.org/3"
IMAGE_URL = "https://image.tmdb.org/t/p/w500"


def search_movie(title):

    if not TMDB_API_KEY:
        raise ValueError("TMDB API key not found in .env")

    url = f"{BASE_URL}/search/movie"

    params = {
        "api_key": TMDB_API_KEY,
        "query": title,
        "include_adult": False,
        "language": "en-US"
    }

    response = requests.get(
        url,
        params=params,
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    results = data.get("results", [])

    if not results:
        return None

    movie = results[0]

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


def get_poster_url(poster_path):

    if not poster_path:
        return None

    return f"{IMAGE_URL}{poster_path}"


def get_backdrop_url(backdrop_path):

    if not backdrop_path:
        return None

    return f"https://image.tmdb.org/t/p/w1280{backdrop_path}"