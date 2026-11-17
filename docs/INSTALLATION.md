# Installation and Setup

## 1. Prerequisites

- Python environment compatible with the pinned project dependencies.
- Git.
- Access to the required datasets and precomputed model artifacts.
- TMDB API key for movie metadata/poster lookup.
- Google Books API key only if running the Google Books enrichment utility.
- A compatible PyTorch installation. CUDA is optional; the LLM module selects CUDA when available and CPU otherwise.

The project was developed in a Windows environment with a Python 3.13 virtual environment. Dependency compatibility can vary by Python version and operating system.

## 2. Clone and create a virtual environment

```bash
git clone <YOUR_REPOSITORY_URL>
cd RecomMind-AI

python -m venv venv
```

Activate on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Activate on macOS/Linux:

```bash
source venv/bin/activate
```

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a PyTorch build appropriate for your operating system and CPU/CUDA setup using the official PyTorch installation selector. Confirm that `torch.cuda.is_available()` returns `True` if GPU inference is expected.

The application imports PyTorch, SciPy, Transformers, and PEFT. Verify that the project's `requirements.txt` includes all direct runtime dependencies for the exact version of the repository being published. In particular, ensure a compatible PyTorch package is installed; do not assume it is installed merely because Transformers is present.

## 4. Configure environment variables

Copy the example file:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and supply your own keys:

```env
TMDB_API_KEY=your_tmdb_api_key
GOOGLE_BOOKS_API_KEY=your_google_books_api_key
```

The TMDB key is used by movie metadata lookup. The Google Books key is used by the metadata enrichment utility. Never commit `.env`.

## 5. Add datasets and model artifacts

A fresh clone does not contain all large datasets and precomputed model artifacts automatically. Populate the expected paths before launching the app.

At minimum, check the following paths based on the active modules:

### Movies
- `datasets/movie_ratings.csv`
- `models/movie_features.csv`
- `models/semantic_movie_features.csv`
- `models/movie_embeddings.npy`
- `models/cosine_similarity.pkl`
- `models/improved_svd_predictions.pkl`
- `models/user_means.pkl`
- `models/improved_svd_movie_ids.npy`
- LightFM export bundle under `models/lightfm/RecomMind_LightFM/`

### Books
- `datasets/books/clean_books.csv`
- `datasets/books/clean_ratings.csv`
- `datasets/books/unified_book_catalog.csv`
- `models/semantic_book_features.csv`
- `models/book_embeddings.npy`

### Products
- `datasets/products/processed/electronics_reviews.csv`
- `datasets/products/processed/electronics_catalog.csv`
- `models/semantic_product_features.csv`
- `models/product_embeddings.npy`

### LLM adapter
- `models/recomind_lora_adapter/` including the adapter configuration, weights, and tokenizer files.

See `DATASETS.md` for provenance and the full artifact overview. Keep related mapping, embedding, feature, and model files from the same build together.

## 6. Run the application

Run from the repository root:

```bash
streamlit run app.py
```

Streamlit will print a local address in the terminal. Open it in your browser.

## 7. Basic troubleshooting

### Missing file errors
Check that the dataset/model artifact exists at the exact path in the error. Preserve the repository directory structure and run Streamlit from the project root.

### Missing API key
Check `.env` spelling and ensure the key is set. Do not paste API keys into issue reports or public logs.

### Model download or initialization takes time
The Sentence Transformer and Qwen model may need to be downloaded the first time unless already cached locally. Product collaborative SVD training also occurs when its recommender module is imported.

### CUDA unavailable
The LLM engine falls back to CPU when CUDA is not available. CPU inference may be slower and memory requirements may differ.

### Incompatible dependency versions
Use a clean virtual environment and install a compatible PyTorch build. If a dependency fails to install on your Python version, use a Python version supported by the pinned package versions and re-test the project.

## 8. Before publishing

- Confirm the app starts from a clean clone.
- Confirm all recommendation modes work with the committed/provisioned artifacts.
- Add dataset source citations and license information.
- Check model and adapter redistribution terms.
- Keep `.env`, raw large datasets, caches, and virtual environments out of Git.
