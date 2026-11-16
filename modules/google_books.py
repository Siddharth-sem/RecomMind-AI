import os
import requests
from dotenv import load_dotenv

load_dotenv()

GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"

API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY")


def search_books(query, max_results=10):

    params = {
        "q": query,
        "maxResults": max_results,
        "printType": "books",
        "key": API_KEY
    }

    try:
        response = requests.get(
            GOOGLE_BOOKS_URL,
            params=params,
            timeout=15
        )

        response.raise_for_status()
        data = response.json()

        results = []

        for item in data.get("items", []):

            info = item.get("volumeInfo", {})

            results.append({
                "google_id": item.get("id"),
                "title": info.get("title", "Unknown"),
                "authors": ", ".join(
                    info.get("authors", ["Unknown"])
                ),
                "description": info.get("description", ""),
                "categories": ", ".join(
                    info.get("categories", [])
                ),
                "published_date": info.get("publishedDate", ""),
                "publisher": info.get("publisher", ""),
                "average_rating": info.get("averageRating"),
                "ratings_count": info.get("ratingsCount", 0),
                "page_count": info.get("pageCount"),
                "thumbnail": info.get(
                    "imageLinks", {}
                ).get("thumbnail"),
                "preview_link": info.get("previewLink")
            })

        return results

    except requests.HTTPError as e:
        print("Google Books HTTP error:", e)
        if e.response is not None:
            print(e.response.text[:1000])
        return []

    except requests.RequestException as e:
        print("Google Books connection error:", e)
        return []


if __name__ == "__main__":

    books = search_books("psychological thriller", 5)

    for book in books:
        print("\nTitle:", book["title"])
        print("Author:", book["authors"])
        print("Published:", book["published_date"])
        print("Description:", book["description"][:200])
        print("Cover:", book["thumbnail"])