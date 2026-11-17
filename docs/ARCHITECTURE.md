# System Architecture

## 1. Overview

RecomMind-AI is a Streamlit application with three recommendation domains: Movies, Books, and Electronics Products. `app.py` provides the user interface, loads recommendation modules, accepts user inputs, invokes the relevant model functions, and renders ranked results.

The application uses a mixture of:
- precomputed model predictions and learned parameters;
- content-based and semantic item representations;
- runtime matrix factorization for Books and Products;
- external metadata APIs for selected display/enrichment tasks;
- a local language model plus LoRA adapter for movie recommendation explanations.

## 2. High-level request flow

```text
User
 |
 v
Streamlit UI (app.py)
 |
 +--> Home / evaluation dashboard
 |
 +--> Movies
 |     +--> Collaborative filtering
 |     +--> Content-based similar movies
 |     +--> Semantic + personalized hybrid retrieval
 |     +--> Hybrid retrieval -> RAG explanation pipeline -> Qwen + LoRA
 |     +--> LightFM WARP inference from exported parameters
 |
 +--> Books
 |     +--> TF-IDF similar-book search
 |     +--> SVD collaborative recommendations
 |     +--> Sentence Transformer semantic search
 |     +--> Hybrid collaborative + semantic ranking
 |
 +--> Products
       +--> SVD collaborative recommendations
       +--> Sentence Transformer semantic search
       +--> Similar-product search
       +--> Hybrid semantic candidate generation + collaborative reranking

Data and artifacts:
  datasets/  -> ratings, cleaned records, product catalog/reviews
  models/    -> embeddings, feature tables, predictions, adapters, LightFM exports
  APIs      -> TMDB movie metadata; Google Books enrichment utility
```

## 3. Application layer

`app.py` defines the navigation and display logic.

- The application has four top-level navigation destinations: Home, Movies, Books, and Products.
- Movie, book, and product cards render domain metadata and available model scores.
- Movie metadata is retrieved through `modules.tmdb_client.search_movie`.
- Book cover selection checks available cover fields in priority order.
- Product cards combine recommendation output with catalog metadata.
- Streamlit `cache_resource` is used for expensive recommendation/model resources so they are not repeatedly initialized on every rerun.

## 4. Movie subsystem

### Collaborative filtering
`modules.movie_recommender` loads MovieLens ratings, movie features, precomputed SVD predictions, user means, and movie ID mappings. For a selected MovieLens user, it ranks predicted ratings and excludes movies already rated by that user.

### Content-based similar movies
The module uses a precomputed content-similarity matrix and movie feature metadata to find titles similar to a selected movie.

### Semantic and personalized hybrid search
The semantic model is `sentence-transformers/all-MiniLM-L6-v2`. The embedding model is loaded lazily and cached. A text query is encoded and combined with personalized retrieval signals in the hybrid recommendation functions.

### LightFM
`modules.movie_lightfm` loads exported arrays, sparse item features, ID mappings, and a movie catalog from `models/lightfm/RecomMind_LightFM/`. The application performs inference from the exported parameters; it does not need to train LightFM at application startup.

### RAG explanations
`modules.rag_pipeline` first retrieves personalized movie candidates. It then asks `modules.llm_engine` for an explanation using the user query and trusted movie metadata. Generated explanations are validated. If generation fails or validation rejects the output, a deterministic fallback explanation is used so recommendation results can still be returned.

The language model is `Qwen/Qwen2.5-0.5B-Instruct` with a local PEFT LoRA adapter at `models/recomind_lora_adapter/`. PyTorch CUDA is selected when available; otherwise CPU is used.

## 5. Book subsystem

- `book_recommender.py` builds TF-IDF vectors from title, author, and publisher, and trains a 50-component TruncatedSVD collaborative model from the cleaned rating matrix.
- `book_semantic.py` loads or builds semantic book embeddings using `all-MiniLM-L6-v2`, then supports natural-language search and similar-book retrieval.
- `book_hybrid.py` retrieves collaborative and semantic candidate lists, merges them by normalized title, scales the two score columns, and combines them using default weights of 0.6 collaborative and 0.4 semantic.
- `book_evaluation.py` contains the book model evaluation workflow.

## 6. Product subsystem

- `product_recommender.py` loads processed electronics reviews, creates a user-product rating matrix, and fits a 50-component TruncatedSVD model. Previously rated products are excluded from collaborative recommendations.
- `product_semantic.py` uses product feature text and `all-MiniLM-L6-v2` embeddings for semantic search and similar-product discovery.
- `product_hybrid.py` uses semantic search to generate a candidate pool, obtains collaborative recommendations, normalizes the two score columns, and ranks candidates using default weights of 0.6 semantic and 0.4 collaborative.
- `product_evaluation.py` contains the product evaluation workflow.

## 7. External services

| Service | Purpose | Configuration |
|---|---|---|
| TMDB API | Movie search and display metadata/posters | `TMDB_API_KEY` |
| Google Books API | Book metadata enrichment utility | `GOOGLE_BOOKS_API_KEY` |

API availability affects metadata enrichment/display, not the core local recommendation artifacts.

## 8. Caching and runtime behavior

- Streamlit caches the movie engine, embedding model, LLM, book engines, and product engines as resources.
- Movie semantic embeddings are loaded through a cached embedding-model function.
- The product collaborative module trains its SVD model when imported, so its first initialization can be comparatively expensive.
- Product hybrid ranking reads catalog data and merges candidate scores at request time.
- The LLM loads once through an `lru_cache`.

## 9. Important path assumptions

Several modules use project-root-relative paths such as `datasets/products/processed/electronics_reviews.csv` and `models/product_embeddings.npy`. Run Streamlit from the repository root and preserve the expected directory layout. Some modules resolve paths from `__file__`, while others use relative paths from the current working directory.
