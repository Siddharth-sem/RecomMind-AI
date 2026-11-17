# RecomMind-AI

**An intelligent recommendation platform for Movies, Books, and Electronics Products.**

RecomMind-AI is a multi-domain recommendation system built with Python and Streamlit. It combines collaborative filtering, content-based similarity, semantic search, hybrid ranking, and a retrieval-augmented generation (RAG) workflow for movie recommendation explanations.

> **Project status:** The Streamlit application and recommendation modes have been implemented and tested by the project author. This repository documentation describes the submitted application and module source. Dataset/model files and API credentials must be supplied separately where noted.

## Features

### Movies
- Collaborative Filtering using precomputed SVD predictions.
- Similar Movies using content similarity.
- Semantic Hybrid Search using a text query and a MovieLens user profile.
- Hybrid + RAG: personalized movie retrieval followed by Qwen + LoRA explanations, with validation and a verified fallback explanation.
- LightFM (WARP) recommendations using exported learned parameters and item features.

### Books
- Semantic Search.
- Collaborative Recommendations.
- Similar Books using TF-IDF features.
- Hybrid Recommendations combining collaborative and semantic scores.
- Book metadata and cover display, with Google Books metadata enrichment supported by a separate utility module.

### Products
- Semantic Product Search.
- Collaborative Recommendations.
- Similar Products.
- Hybrid Recommendations combining semantic and collaborative scores.
- Electronics product metadata including title, brand/store, price (when available), ratings, categories, descriptions, and image URLs.

### Evaluation dashboard
The Home page displays separate evaluation summaries for Movies, Books, and Products. Results are reported within each domain; raw metrics should **not** be compared across domains because the datasets, relevance thresholds, and evaluation protocols differ.

## Technology stack

- Python
- Streamlit
- pandas, NumPy, SciPy
- scikit-learn
- Sentence Transformers (`sentence-transformers/all-MiniLM-L6-v2`)
- Transformers and PEFT
- PyTorch
- Matplotlib
- Requests and python-dotenv

## Repository layout

```text
RecomMind-AI/
├── app.py
├── modules/
│   ├── movie_recommender.py
│   ├── movie_knn.py
│   ├── movie_nmf.py
│   ├── movie_lightfm.py
│   ├── rag_pipeline.py
│   ├── llm_engine.py
│   ├── book_recommender.py
│   ├── book_semantic.py
│   ├── book_hybrid.py
│   ├── book_evaluation.py
│   ├── product_recommender.py
│   ├── product_semantic.py
│   ├── product_hybrid.py
│   ├── product_evaluation.py
│   ├── tmdb_client.py
│   └── google_books.py
├── datasets/
├── models/
├── evaluation_results/
├── docs/
├── requirements.txt
├── .env.example
└── .gitignore
```

The exact set of generated model files and datasets may vary depending on how the project artifacts are provisioned. See [Datasets](docs/DATASETS.md), [Installation](docs/INSTALLATION.md), and [Architecture](docs/ARCHITECTURE.md).

## Quick start

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies. Install a compatible PyTorch build for your system separately if it is not provided by the dependency installation.
4. Copy `.env.example` to `.env` and add your own API keys.
5. Place the required dataset and model artifacts in the paths described in the documentation.
6. Start Streamlit:

```bash
streamlit run app.py
```

For detailed setup instructions and required files, read [docs/INSTALLATION.md](docs/INSTALLATION.md).

## Evaluation snapshot

The following values are the results currently displayed by the application's evaluation dashboard.

| Domain | Evaluated approach | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---:|---:|---:|
| Movies | User KNN | 16.61% | 17.47% | 23.15% |
| Movies | Item KNN | 18.69% | 20.14% | 25.48% |
| Movies | NMF | 18.62% | 19.77% | 25.30% |
| Movies | LightFM (WARP) | **22.21%** | **22.66%** | **28.96%** |
| Books | Collaborative Filtering | 1.68% | 5.41% | 4.22% |
| Books | Semantic Recommendations | 0.75% | 3.51% | 1.98% |
| Books | Hybrid Recommendations | **1.93%** | **6.70%** | **4.90%** |
| Products | Collaborative | 0.96% | 3.53% | 2.37% |
| Products | Semantic | 0.75% | 2.93% | 1.80% |
| Products | Hybrid | **1.15%** | **4.31%** | **2.93%** |

**Interpretation:** LightFM has the strongest displayed ranking metrics among the evaluated movie models. Hybrid is strongest among the evaluated book methods and among the evaluated product methods. NMF has the highest displayed movie catalog coverage (25.98%). See [Evaluation methodology and limitations](docs/EVALUATION.md).

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Technical documentation](docs/TECHNICAL_DOCUMENTATION.md)
- [Datasets and model artifacts](docs/DATASETS.md)
- [Evaluation](docs/EVALUATION.md)
- [Installation and setup](docs/INSTALLATION.md)

## Important repository hygiene

The raw Amazon Electronics metadata file is approximately 5.2 GB and the raw review file is approximately 760 MB. Do not commit these large raw files to a normal GitHub repository. Keep API keys, virtual environments, caches, and generated temporary files out of version control. Review the size and redistribution terms of model weights, adapters, datasets, and other binary artifacts before publishing them.

## Limitations

- Recommendations depend on the availability and quality of the supplied datasets and precomputed model artifacts.
- Collaborative filtering has limited ability to recommend for new users or unseen items without interaction history.
- Semantic results depend on the text/features used to build the embeddings.
- Product prices, images, and metadata may be missing or stale in the source catalog.
- The RAG explanation workflow is limited to the trusted metadata passed into the prompt; generated text is validated and replaced with a fallback when validation fails.
- The app's model and data files are loaded from local paths. A fresh clone is not self-contained until the required artifacts are supplied.

## License

No license has been specified in the supplied project files. Add a license only after deciding how the source code, datasets, and model artifacts may be reused.
