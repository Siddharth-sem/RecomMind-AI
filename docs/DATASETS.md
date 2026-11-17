# Datasets and Model Artifacts

## 1. Movie data

The project uses the MovieLens 100K dataset.

Reported dataset summary:
- 100,000 ratings
- 943 unique users
- 1,682 unique movies

Project-level files include:
- `datasets/movie_ratings.csv`
- `datasets/movie_genres.csv`

Movie feature tables, embeddings, similarity matrices, prediction arrays, and other trained artifacts are stored under `models/`. The LightFM inference bundle is stored under:

```text
models/lightfm/RecomMind_LightFM/
├── evaluation_results.csv
├── item_features.npz
├── lightfm_parameters.npz
├── mappings.json
└── movie_catalog.csv
```

The LightFM files must be kept together because their user/item mappings, feature rows, and learned parameters must remain aligned.

## 2. Book data

The book subsystem is based on Book-Crossing data and a metadata-enriched catalog.

Reported prepared-data counts:
- 10,525 filtered books in the cleaned Book-Crossing set;
- 134,046 ratings;
- 11,969 users;
- 10,667 entries in the unified book catalog.

Files reported in the project:
- `datasets/books/BX-Book-Ratings.csv`
- `datasets/books/BX-Books.csv`
- `datasets/books/BX-Users.csv`
- `datasets/books/clean_books.csv`
- `datasets/books/clean_ratings.csv`
- `datasets/books/google_books_catalog.csv`
- `datasets/books/unified_book_catalog.csv`
- `datasets/books/book_evaluation_results.csv`

Book embedding and semantic feature artifacts:
- `models/book_embeddings.npy`
- `models/semantic_book_features.csv`

Book cover URLs and descriptive metadata are included where available. Google Books enrichment is implemented in `modules/google_books.py` and `modules/build_google_books_catalog.py`.

## 3. Product data

The product subsystem uses the Amazon Reviews 2023 Electronics 5-core subset.

Reported dataset summary:
- 1,172,576 interactions;
- 162,812 users;
- 2,000 products;
- approximately 71.7% five-star interactions;
- 570 of 2,000 products have missing prices.

Processed files:
- `datasets/products/processed/electronics_reviews.csv`
- `datasets/products/processed/electronics_catalog.csv`

The catalog includes:
`parent_asin`, `title`, `brand`, `store`, `main_category`, `categories`, `description`, `features`, `price`, `average_rating`, `rating_number`, and `image_url`.

Raw files previously reported:
- `datasets/products/raw/electronics_metadata.jsonl` (~5.2 GB)
- `datasets/products/raw/electronics_reviews.csv` (~760 MB)

**Do not commit these raw files to GitHub.** Provide dataset acquisition/preparation instructions or use an appropriate external dataset host if redistribution is permitted. Confirm the dataset's license and terms before publishing or sharing it.

Product model artifacts include:
- `models/semantic_product_features.csv`
- `models/product_embeddings.npy`

## 4. Shared model artifacts

The project also contains movie artifacts such as:
- `models/movie_features.csv`
- `models/semantic_movie_features.csv`
- `models/movie_embeddings.npy`
- `models/cosine_similarity.pkl`
- `models/improved_svd_predictions.pkl`
- `models/improved_svd_movie_ids.npy`
- `models/user_means.pkl`

The LoRA adapter directory is:
`models/recomind_lora_adapter/`

It contains adapter configuration, tokenizer files, and model weights. Check the size and model licensing terms before publishing it.

## 5. Reproducibility and alignment

Recommendation artifacts are not interchangeable:
- embedding rows must align with their feature/catalog rows;
- matrix-factorization predictions must align with their saved user/item mappings;
- LightFM parameters must align with the supplied mapping and item-feature files;
- the model artifacts must correspond to the same dataset preprocessing version.

If an artifact is missing, use the project's preparation/training scripts where available. Do not create placeholder arrays or substitute a different dataset without updating the mappings and evaluation.

## 6. Data provenance and licensing

This documentation records the dataset names and project-level counts supplied during development. Before public release, include the original dataset citations, license/terms, and any required attribution. No dataset license was established from the supplied project files, so this document does not grant redistribution rights.
