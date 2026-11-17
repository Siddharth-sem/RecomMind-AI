# Technical Documentation

## 1. Application entry point

`app.py` is the Streamlit entry point. It:
1. configures the page and global styling;
2. renders the RecomMind-AI brand header and top navigation;
3. selects the active page from Streamlit session state;
4. initializes recommendation engines through cached loader functions;
5. collects user inputs and calls module-level recommendation functions;
6. displays item metadata, model scores, and evaluation charts.

The four top-level pages are Home, Movies, Books, and Products.

## 2. Movie recommendation methods

### 2.1 Collaborative Filtering
Function: `get_collaborative_recommendations(user_id, top_n=10)` in `modules/movie_recommender.py`.

The engine reads stored SVD predictions and maps them back to movie IDs. It validates the user ID, removes movies already rated by the user, sorts by predicted rating, and joins title/genre metadata.

### 2.2 Similar Movies
Function: `get_content_recommendations(movie_title, top_n=10)`.

Uses the precomputed content similarity matrix and movie features to rank similar titles. The selected title is not returned as its own recommendation.

### 2.3 Semantic Hybrid Search
Function: `get_personalized_hybrid_recommendations(user_id, query, top_n)`.

Uses a natural-language preference query and a MovieLens user profile to produce personalized recommendations. Text embeddings use `sentence-transformers/all-MiniLM-L6-v2`.

### 2.4 Hybrid + RAG
Function: `generate_rag_recommendations(user_id, query, top_n=5)` in `modules/rag_pipeline.py`.

Pipeline:
1. Validate the query and requested result count.
2. Retrieve personalized movie candidates from trusted project data.
3. Pass the query, title, genres, and average rating to the explanation generator.
4. Validate the generated explanation for empty/overlong output, malformed content, title relevance, and unsupported plot/production claims.
5. Use a verified fallback explanation if generation fails or validation rejects the output.
6. Return the recommendation with its score and explanation source.

The LLM prompt instructs the model to use only supplied metadata and not invent ratings, prices, features, specifications, or plot details. The movie explanation currently uses genre and average rating metadata.

### 2.5 LightFM (WARP)
Function: `get_lightfm_recommendations(user_id, top_n=10)` in `modules/movie_lightfm.py`.

The runtime module loads exported user/item embeddings and biases, item-feature data, mappings, and the movie catalog. The saved model was trained/exported separately; local recommendation inference reconstructs scoring from the saved arrays rather than importing the LightFM training package.

## 3. Book recommendation methods

### 3.1 Collaborative Recommendations
`modules/book_recommender.py` creates a user-by-ISBN rating matrix and applies `TruncatedSVD(n_components=50, random_state=42)`. Recommendations are sorted by predicted rating and exclude ISBNs already rated by the selected user.

### 3.2 Similar Books
The same module builds TF-IDF features from:
- book title;
- author;
- publisher.

Configuration: English stop words, unigrams and bigrams, and a maximum of 20,000 features. Cosine similarity is used to retrieve similar titles.

### 3.3 Semantic Search
`modules.book_semantic.py` uses `sentence-transformers/all-MiniLM-L6-v2` embeddings over the book feature representation. It supports natural-language search and similar-book retrieval.

### 3.4 Hybrid Recommendations
`modules.book_hybrid.py` obtains collaborative recommendations and semantically similar books, merges the result sets using normalized title text, independently scales the two scores, and calculates a weighted score.

Default weights:
- Collaborative: 0.6
- Semantic: 0.4

## 4. Product recommendation methods

### 4.1 Collaborative Recommendations
`modules/product_recommender.py` pivots processed review data into a user-by-product matrix, fits `TruncatedSVD(n_components=50, random_state=42)`, reconstructs predicted ratings, excludes previously rated products, and joins the electronics catalog.

### 4.2 Semantic Product Search and Similar Products
`modules/product_semantic.py` loads product feature data and precomputed embeddings. It uses the `all-MiniLM-L6-v2` Sentence Transformer for query embedding and cosine similarity ranking.

### 4.3 Hybrid Recommendations
`modules/product_hybrid.py`:
1. generates up to 100 semantic candidates;
2. generates up to 200 collaborative recommendations;
3. joins collaborative scores onto the semantic candidate pool;
4. fills missing collaborative scores with zero;
5. min-max normalizes the two score columns;
6. combines them with default weights of 0.6 semantic and 0.4 collaborative;
7. excludes products already rated by the user;
8. joins catalog metadata and returns the highest-scoring results.

## 5. Model and embedding artifacts

The application expects precomputed artifacts in `models/`, including movie predictions/features, semantic feature tables and embeddings, book embeddings, product embeddings, LightFM exports, and a LoRA adapter directory.

Do not assume these artifacts are regenerated automatically by `streamlit run app.py`. Refer to `DATASETS.md` for the expected files and `INSTALLATION.md` for setup checks.

## 6. API integrations

### TMDB
`modules/tmdb_client.py` reads `TMDB_API_KEY` from `.env`. It searches by title, optionally extracts a year from a MovieLens title, and returns metadata such as release date, overview, rating, and poster/backdrop URLs.

### Google Books
`modules/google_books.py` reads `GOOGLE_BOOKS_API_KEY` and provides a search utility that returns book title, authors, description, categories, publication date, publisher, ratings, page count, thumbnail, preview link, and industry identifiers. This utility is used for metadata enrichment workflows; the Streamlit book page displays the locally prepared catalog.

## 7. Error handling

The UI wraps recommendation calls in exception handlers and displays a method-specific error message. Book and product page initialization also catches engine-loading errors and displays the expected data/artifact paths. The RAG pipeline is designed to preserve recommendations if explanation generation fails by using a fallback explanation.

## 8. Runtime considerations

- The first call to semantic search may load the Sentence Transformer model.
- The product collaborative recommender trains SVD when the module is imported; initial product-page loading may therefore take longer.
- Qwen + LoRA model loading is cached but can require substantial memory. CUDA is used when available; CPU inference is the fallback.
- Model/data files must match the mappings and feature order used when they were created. Replacing one artifact independently can produce incorrect recommendations or shape/mapping errors.
