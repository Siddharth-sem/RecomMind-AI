import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import textwrap
from modules.tmdb_client import search_movie

@st.cache_resource(show_spinner="Loading movie engine...")
def load_movie_engine():
    from modules import movie_recommender
    return movie_recommender


@st.cache_resource(show_spinner="Loading semantic model...")
def load_embedding_model():
    from modules.movie_recommender import get_embedding_model
    return get_embedding_model()


@st.cache_resource(show_spinner="Loading Qwen + LoRA...")
def load_ai_model():
    from modules.llm_engine import load_llm
    return load_llm()

from modules.movie_recommender import (
    get_collaborative_recommendations,
    get_content_recommendations,
    get_personalized_hybrid_recommendations,
    get_lightfm_recommendations,
    get_engine_info
)

from modules.rag_pipeline import generate_rag_recommendations


@st.cache_resource(show_spinner="Loading book recommendation engines...")
def load_book_engines():
    from modules import book_recommender, book_semantic, book_hybrid

    return {
        "recommender": book_recommender,
        "semantic": book_semantic,
        "hybrid": book_hybrid,
    }


def display_book_card(book):
    """Display book details with cover image and recommendation scores."""

    title = str(book.get("Book-Title", "Untitled book"))
    author = str(book.get("Book-Author", "Unknown author"))

    publisher = book.get("Publisher", "")
    description = book.get("description", "")
    categories = book.get("categories", "")

    # Find the best available cover image
    image_url = None

    image_columns = [
        "Image-URL-L",
        "Image-URL-M",
        "thumbnail",
        "Image-URL-S"
    ]

    for column in image_columns:
        value = book.get(column)

        if (
            isinstance(value, str)
            and value.strip()
            and value.strip().lower() not in ["nan", "none"]
            and value.strip().startswith(("http://", "https://"))
        ):
            image_url = value.strip()
            break

    # Create layout
    col1, col2 = st.columns([1, 4])

    # Book cover
    with col1:
        if image_url:
            st.image(image_url, width=150)
        else:
            st.markdown("📚 Cover unavailable")

    # Book information
    with col2:
        st.subheader(title)
        st.caption(f"✍️ Author: {author}")

        if pd.notna(publisher) and str(publisher).strip():
            st.caption(f"Publisher: {publisher}")

        if pd.notna(categories) and str(categories).strip():
            st.caption(f"Categories: {categories}")

        # Recommendation scores
        if pd.notna(book.get("predicted_rating")):
            st.write(
                f"⭐ Collaborative Score: "
                f"{float(book['predicted_rating']):.3f}"
            )

        if pd.notna(book.get("semantic_similarity")):
            st.write(
                f"🧠 Semantic Similarity: "
                f"{float(book['semantic_similarity']):.3f}"
            )

        if pd.notna(book.get("similarity")):
            st.write(
                f"🔎 Content Similarity: "
                f"{float(book['similarity']):.3f}"
            )

        if pd.notna(book.get("hybrid_score")):
            st.write(
                f"🎯 Hybrid Score: "
                f"{float(book['hybrid_score']):.3f}"
            )

        # Description
        if (
            isinstance(description, str)
            and description.strip()
            and description.strip().lower() != "nan"
        ):
            st.write(
                description[:650] +
                ("..." if len(description) > 650 else "")
            )

    st.divider()





# ==========================================
# PRODUCT RECOMMENDATION HELPERS
# ==========================================

@st.cache_resource(show_spinner="Loading product recommendation engines...")
def load_product_engines():
    # These imports are intentionally lazy. The product collaborative engine
    # trains its SVD model when imported, so Streamlit should do that only once.
    from modules import product_recommender, product_semantic, product_hybrid

    catalog = pd.read_csv(
        "datasets/products/processed/electronics_catalog.csv"
    )

    return {
        "recommender": product_recommender,
        "semantic": product_semantic,
        "hybrid": product_hybrid,
        "catalog": catalog,
    }


def display_product_card(product, catalog):
    """Render a product with catalog metadata and available model scores."""
    item = product.to_dict() if hasattr(product, "to_dict") else dict(product)

    asin = item.get("parent_asin")
    if pd.notna(asin):
        matched = catalog.loc[catalog["parent_asin"] == asin]
        if not matched.empty:
            # Catalog metadata is the fallback; recommendation-result values
            # take precedence when they are populated.
            metadata = matched.iloc[0].to_dict()
            for key, value in item.items():
                if value is not None and not (
                    isinstance(value, float) and pd.isna(value)
                ):
                    metadata[key] = value
            item = metadata

    title = item.get("title")
    if not isinstance(title, str) or not title.strip():
        title = "Untitled product"

    image_url = item.get("image_url")
    valid_image = (
        isinstance(image_url, str)
        and image_url.strip().startswith(("http://", "https://"))
        and image_url.strip().lower() not in ("nan", "none")
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        if valid_image:
            st.image(image_url.strip(), width=150)
        else:
            st.markdown("🛍️ Image unavailable")

    with col2:
        st.subheader(title)

        brand = item.get("brand")
        store = item.get("store")
        if isinstance(brand, str) and brand.strip() and brand.lower() != "nan":
            st.caption(f"Brand: {brand}")
        elif isinstance(store, str) and store.strip() and store.lower() != "nan":
            st.caption(f"Store: {store}")

        price = item.get("price")
        if pd.notna(price):
            try:
                st.markdown(f"**Price: ${float(price):,.2f}**")
            except (TypeError, ValueError):
                st.markdown(f"**Price: {price}**")

        rating = item.get("average_rating")
        rating_count = item.get("rating_number")
        if pd.notna(rating):
            rating_text = f"⭐ {float(rating):.1f}/5"
            if pd.notna(rating_count):
                rating_text += f" ({int(float(rating_count)):,} ratings)"
            st.caption(rating_text)

        category = item.get("categories")
        if isinstance(category, str) and category.strip() and category.lower() != "nan":
            st.caption(f"Category: {category[:220]}")

        for key, label in [
            ("collaborative_score", "Collaborative Score"),
            ("semantic_similarity", "Semantic Similarity"),
            ("hybrid_score", "Hybrid Score"),
        ]:
            value = item.get(key)
            if pd.notna(value):
                try:
                    st.write(f"**{label}:** {float(value):.4f}")
                except (TypeError, ValueError):
                    pass

        description = item.get("description")
        if isinstance(description, str) and description.strip() and description.lower() != "nan":
            st.write(description[:650] + ("..." if len(description) > 650 else ""))

    st.divider()


def display_product_results(results, catalog):
    if results is None or results.empty:
        st.info("No matching products were found.")
        return

    st.success(f"Found {len(results)} product recommendations.")
    for _, product in results.iterrows():
        display_product_card(product, catalog)


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="RecomMind-AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)
@st.cache_data(show_spinner=False)
def get_movie_details(title):
    try:
        return search_movie(title)
    except Exception:
        return None


def display_movie_card(movie, explanation=None):

    details = get_movie_details(movie["title"])

    col1, col2 = st.columns([1, 3])

    with col1:
        if details and details.get("poster_url"):
            st.image(
                details["poster_url"],
                width="stretch"
            )
        else:
            st.markdown("🎬 Poster unavailable")

    with col2:
        st.subheader(movie["title"])

        if details:
            st.caption(
                f"Release Date: {details.get('release_date', 'Unknown')} "
                f"| TMDB Rating: {details.get('rating', 'N/A')}"
            )

            if details.get("overview"):
                st.write(details["overview"])

        genres = movie.get("genre_text", movie.get("genres", ""))

        if genres:
            st.caption(f"Genres: {genres}")

        if movie.get("predicted_rating") is not None:
            st.write(
                f"⭐ Predicted Rating: "
                f"{movie['predicted_rating']:.2f}/5"
            )

        if movie.get("semantic_similarity") is not None:
            st.write(
                f"🧠 Semantic Similarity: "
                f"{movie['semantic_similarity']:.3f}"
            )

        if movie.get("hybrid_score") is not None:
            st.write(
                f"🎯 Hybrid Score: "
                f"{movie['hybrid_score']:.3f}"
            )

        if movie.get("lightfm_score") is not None:
            st.write(
                f"⚡ LightFM Ranking Score: "
                f"{movie['lightfm_score']:.3f}"
            )

        if explanation:
            st.markdown("**🤖 AI Explanation**")
            st.write(explanation)

    st.divider()


# INITIALIZE ENGINE HERE
load_movie_engine()

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0b1020;
    color: #f1f5f9;
}

.block-container {
    padding-top: 2.5rem;
    max-width: 1200px;
}

.main-title {
    font-size: 44px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -1.5px;
}

.subtitle {
    color: #94a3b8;
    font-size: 17px;
    margin-top: 8px;
}

.movie-card {
    background: linear-gradient(145deg, #172033, #111827);
    padding: 22px;
    border-radius: 16px;
    margin-bottom: 18px;
    border: 1px solid #263449;
    transition: 0.2s ease;
}

.movie-card:hover {
    border-color: #6366f1;
    transform: translateY(-2px);
}

.movie-title {
    font-size: 21px;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 10px;
}

.movie-meta {
    color: #a5b4c8;
    font-size: 14px;
    margin-bottom: 6px;
}

.movie-explanation {
    color: #dbeafe;
    line-height: 1.7;
    font-size: 15px;
}

div[data-testid="stMetric"] {
    background: #172033;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #263449;
}

div[data-testid="stMetricLabel"] {
    color: #94a3b8;
}

div[data-testid="stMetricValue"] {
    color: #ffffff;
}

.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    min-height: 42px;
}

section[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #263449;
}

hr {
    border-color: #263449;
}

.movie-badge {
    display: inline-block;
    background: #263653;
    color: #bfdbfe;
    padding: 5px 11px;
    border-radius: 20px;
    font-size: 12px;
    margin: 5px 5px 5px 0;
}

.movie-score {
    color: #a5b4fc;
    font-size: 14px;
    font-weight: 600;
    margin-top: 10px;
}

.movie-card {
    background: linear-gradient(145deg, #172033, #111827);
    padding: 22px;
    border-radius: 16px;
    margin-bottom: 18px;
    border: 1px solid #263449;
    transition: 0.2s ease;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# GLOBAL BRAND HEADER
# ==========================================

st.markdown("""
<style>
.brand-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 4px 0 18px 0;
    margin-bottom: 14px;
    border-bottom: 1px solid #263449;
}
.brand-mark {
    width: 48px;
    height: 48px;
    flex: 0 0 48px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 14px;
    background: linear-gradient(135deg, #6366f1, #0ea5e9);
    color: white;
    font-size: 25px;
    font-weight: 800;
    box-shadow: 0 8px 22px rgba(79, 70, 229, 0.22);
}
.brand-name {
    color: #f8fafc;
    font-size: 27px;
    line-height: 1.1;
    font-weight: 800;
    letter-spacing: -0.8px;
}
.brand-name span {
    color: #a5b4fc;
}
.brand-tagline {
    color: #94a3b8;
    font-size: 12px;
    margin-top: 4px;
    letter-spacing: 0.3px;
}
@media (max-width: 600px) {
    .brand-header { gap: 10px; padding-bottom: 13px; }
    .brand-mark { width: 42px; height: 42px; flex-basis: 42px; }
    .brand-name { font-size: 22px; }
}
</style>
<div class="brand-header">
    <div class="brand-mark">R</div>
    <div>
        <div class="brand-name">RecomMind<span>-AI</span></div>
        <div class="brand-tagline">INTELLIGENT RECOMMENDATION PLATFORM</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# TOP TILE NAVIGATION
# ==========================================

if "active_page" not in st.session_state:
    st.session_state.active_page = "Home"

st.markdown("""
<style>
.top-nav-tile button {
    width: 100%;
    min-height: 76px;
    border-radius: 14px;
    border: 1px solid #334155;
    background: linear-gradient(145deg, #172033, #111827);
    color: #e2e8f0;
    font-size: 17px;
    font-weight: 700;
    transition: all 0.18s ease;
}
.top-nav-tile button:hover {
    border-color: #818cf8;
    color: #ffffff;
    background: linear-gradient(145deg, #263653, #172033);
}
.top-nav-tile button:focus {
    border: 1px solid #818cf8;
    color: #ffffff;
}
.top-nav-caption {
    color: #94a3b8;
    font-size: 12px;
    text-align: center;
    margin-top: -8px;
}
</style>
""", unsafe_allow_html=True)

nav_items = [
    ("🏠  HOME", "Home"),
    ("🎬  MOVIES", "Movies"),
    ("📚  BOOKS", "Books"),
    ("🛍️  PRODUCTS", "Products"),
]

nav_columns = st.columns(4, gap="medium")
for column, (label, page_name) in zip(nav_columns, nav_items):
    with column:
        st.markdown('<div class="top-nav-tile">', unsafe_allow_html=True)
        if st.button(
            label,
            key=f"nav_{page_name.lower()}",
            type="primary" if st.session_state.active_page == page_name else "secondary",
            width="stretch",
        ):
            st.session_state.active_page = page_name
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
page = st.session_state.active_page
st.divider()


# ==========================================
# HOME PAGE
# ==========================================

if page == "Home":

    st.markdown(
        '<div class="main-title">Welcome to RecomMind AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Discover movies tailored to your interests using Artificial Intelligence.'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2, col3 = st.columns(3)

    info = get_engine_info()

    col1.metric("Total Movies", info["movies"])
    col2.metric("Total Users", info["users"])
    col3.metric("Total Ratings", f'{info["ratings"]:,}')

    st.divider()

    st.subheader("What makes RecomMind different?")

    a, b = st.columns(2)

    with a:
        st.markdown("""
        ### 🎯 Personalized Recommendations

        Recommendations based on user rating history
        and individual preferences.
        """)

    with b:
        st.markdown("""
        ### 🧠 AI-Powered Explanations

        Qwen + LoRA generates explanations
        for recommended movies using retrieved metadata.
        """)

    st.info(
        "Use the sidebar to discover movies or generate "
        "personalized recommendations."
    )

    st.divider()

    st.subheader("📊 Unified Movie Recommendation Evaluation")
    st.caption(
        "Four algorithms evaluated on the same positive-interaction train/test split "
        "using Precision@10, Recall@10, NDCG@10 and catalog coverage."
    )

    evaluation_data = pd.DataFrame({
        "Algorithm": ["User KNN", "Item KNN", "NMF", "LightFM (WARP)"],
        "Precision@10": [0.1661359, 0.1869427, 0.1861996, 0.2220807],
        "Recall@10": [0.1746507, 0.2013573, 0.1976627, 0.2265795],
        "NDCG@10": [0.2314591, 0.2548250, 0.2529811, 0.2896272],
        "Catalog Coverage": [0.0434007, 0.1087990, 0.2598098, 0.2378121]
    })

    best = evaluation_data.loc[evaluation_data["NDCG@10"].idxmax()]
    metric_cols = st.columns(4)
    metric_cols[0].metric("Best Precision@10", f"{evaluation_data['Precision@10'].max():.2%}", "LightFM (WARP)")
    metric_cols[1].metric("Best Recall@10", f"{evaluation_data['Recall@10'].max():.2%}", "LightFM (WARP)")
    metric_cols[2].metric("Best NDCG@10", f"{best['NDCG@10']:.2%}", "LightFM (WARP)")
    metric_cols[3].metric("Best Catalog Coverage", f"{evaluation_data['Catalog Coverage'].max():.2%}", "NMF")

    st.markdown("### Ranking quality comparison")
    chart_data = evaluation_data.set_index("Algorithm")[[
        "Precision@10", "Recall@10", "NDCG@10"
    ]]
    fig, ax = plt.subplots(figsize=(11, 5))
    chart_data.plot(kind="bar", ax=ax, width=0.75)
    ax.set_ylabel("Evaluation score")
    ax.set_xlabel("")
    ax.set_ylim(0, 0.34)
    ax.set_title("Ranking Metrics at K=10")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Metric")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Catalog coverage")
    coverage_data = evaluation_data.set_index("Algorithm")[["Catalog Coverage"]]
    fig, ax = plt.subplots(figsize=(11, 3.5))
    coverage_data.plot(kind="bar", ax=ax, legend=False, width=0.65)
    ax.set_ylabel("Fraction of catalog recommended")
    ax.set_xlabel("")
    ax.set_ylim(0, 0.32)
    ax.set_title("Catalog Coverage")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    display_data = evaluation_data.copy()
    for column in ["Precision@10", "Recall@10", "NDCG@10", "Catalog Coverage"]:
        display_data[column] = display_data[column].map(lambda value: f"{value:.2%}")
    st.markdown("### Exact evaluation results")
    st.dataframe(display_data, hide_index=True, width="stretch")

    st.success(
        "LightFM (WARP) achieves the strongest ranking quality across all three "
        "ranking metrics, while NMF recommends the largest fraction of the catalog."
    )
    st.caption(
        "Methodology: common positive-only evaluation split; relevance threshold ≥4; "
        "top-10 recommendations; previously seen items excluded. These results are "
        "separate from the earlier Hybrid + RAG evaluation, which used a different protocol."
    )


    # ==========================================
    # BOOK EVALUATION
    # ==========================================

    st.divider()
    st.subheader("📚 Book Recommendation Evaluation")
    st.caption(
        "Book algorithms evaluated on the Book-Crossing dataset. "
        "These results use a separate evaluation protocol from Movies and Products."
    )

    book_evaluation = pd.DataFrame({
        "Algorithm": ["Collaborative Filtering", "Semantic Search", "Hybrid"],
        "Precision@10": [0.0168, 0.0075, 0.0193],
        "Recall@10": [0.0541, 0.0351, 0.0670],
        "NDCG@10": [0.0422, 0.0198, 0.0490],
    })

    book_best = book_evaluation.loc[book_evaluation["NDCG@10"].idxmax()]
    book_metrics = st.columns(3)
    book_metrics[0].metric(
        "Best Precision@10",
        f"{book_evaluation['Precision@10'].max():.2%}",
        "Hybrid",
    )
    book_metrics[1].metric(
        "Best Recall@10",
        f"{book_evaluation['Recall@10'].max():.2%}",
        "Hybrid",
    )
    book_metrics[2].metric(
        "Best NDCG@10",
        f"{book_best['NDCG@10']:.2%}",
        "Hybrid",
    )

    book_chart = book_evaluation.set_index("Algorithm")[
        ["Precision@10", "Recall@10", "NDCG@10"]
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    book_chart.plot(kind="bar", ax=ax, width=0.7)
    ax.set_title("Book Recommendation Ranking Metrics")
    ax.set_ylabel("Evaluation score")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Metric")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    book_display = book_evaluation.copy()
    for column in ["Precision@10", "Recall@10", "NDCG@10"]:
        book_display[column] = book_display[column].map(lambda value: f"{value:.2%}")
    st.dataframe(book_display, hide_index=True, width="stretch")
    st.success("Hybrid is the strongest of the three evaluated book methods.")

    # ==========================================
    # PRODUCT EVALUATION
    # ==========================================

    st.divider()
    st.subheader("🛍️ Product Recommendation Evaluation")
    st.caption(
        "Electronics recommendation evaluation. Product metrics are reported "
        "separately because the dataset and evaluation setup differ from the other domains."
    )

    product_evaluation = pd.DataFrame({
        "Algorithm": ["Collaborative Filtering", "Semantic Search", "Hybrid"],
        "Precision@10": [0.009582, 0.007543, 0.011519],
        "Recall@10": [0.035264, 0.029307, 0.043054],
        "NDCG@10": [0.023669, 0.017997, 0.029310],
    })

    product_best = product_evaluation.loc[
        product_evaluation["NDCG@10"].idxmax()
    ]
    product_metrics = st.columns(3)
    product_metrics[0].metric(
        "Best Precision@10",
        f"{product_evaluation['Precision@10'].max():.2%}",
        "Hybrid",
    )
    product_metrics[1].metric(
        "Best Recall@10",
        f"{product_evaluation['Recall@10'].max():.2%}",
        "Hybrid",
    )
    product_metrics[2].metric(
        "Best NDCG@10",
        f"{product_best['NDCG@10']:.2%}",
        "Hybrid",
    )

    product_chart = product_evaluation.set_index("Algorithm")[
        ["Precision@10", "Recall@10", "NDCG@10"]
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    product_chart.plot(kind="bar", ax=ax, width=0.7)
    ax.set_title("Product Recommendation Ranking Metrics")
    ax.set_ylabel("Evaluation score")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Metric")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    product_display = product_evaluation.copy()
    for column in ["Precision@10", "Recall@10", "NDCG@10"]:
        product_display[column] = product_display[column].map(
            lambda value: f"{value:.2%}"
        )
    st.dataframe(product_display, hide_index=True, width="stretch")
    st.success("Hybrid is the strongest of the three evaluated product methods.")

    st.info(
        "Interpretation: compare algorithms within the same domain. "
        "Do not rank Movies, Books and Products against one another using these raw "
        "metric values because their datasets, relevance thresholds and evaluation "
        "protocols are not identical."
    )


# ==========================================
# MOVIES — DISCOVERY & PERSONALIZED RECOMMENDATIONS
# ==========================================

elif page == "Movies":

    st.title("🎬 Movies")
    st.markdown(
        '<div class="subtitle">Discover movies, explore similar titles, '
        'and get personalized recommendations powered by AI.</div>',
        unsafe_allow_html=True,
    )

    recommendation_mode = st.selectbox(
        "Choose a recommendation method",
        [
            "Collaborative Filtering",
            "Similar Movies",
            "Semantic Hybrid Search",
            "Hybrid + RAG (AI Explanations)",
            "LightFM (WARP)",
        ],
        key="movies_recommendation_mode",
    )

    st.divider()

    if recommendation_mode == "Collaborative Filtering":
        st.caption("Recommendations based on MovieLens user-rating patterns.")
        user_id = st.number_input(
            "MovieLens User ID", min_value=1, max_value=943, value=1,
            key="movies_cf_user",
        )
        top_n = st.slider(
            "Number of recommendations", 5, 20, 10, key="movies_cf_top_n"
        )

        if st.button("Recommend Movies", type="primary", width="stretch"):
            try:
                with st.spinner("Finding recommendations..."):
                    results = get_collaborative_recommendations(int(user_id), top_n)
                if results.empty:
                    st.info("No recommendations found.")
                else:
                    st.success("Recommendations generated!")
                    for _, movie in results.iterrows():
                        display_movie_card(movie)
            except Exception as error:
                st.error(f"Collaborative recommendations failed: {error}")

    elif recommendation_mode == "Similar Movies":
        movie_options = sorted(
            pd.Series(
                __import__(
                    "modules.movie_recommender",
                    fromlist=["movie_features"],
                ).movie_features["title"]
            ).dropna().unique().tolist()
        )
        selected_movie = st.selectbox(
            "Select a movie", movie_options, key="movies_similar_title"
        )
        top_n = st.slider(
            "Number of similar movies", 5, 20, 10,
            key="movies_similar_top_n",
        )

        if st.button("Find Similar Movies", type="primary", width="stretch"):
            try:
                with st.spinner("Finding similar movies..."):
                    results = get_content_recommendations(selected_movie, top_n)
                if results.empty:
                    st.info("No similar movies found.")
                else:
                    for _, movie in results.iterrows():
                        display_movie_card(movie)
            except Exception as error:
                st.error(f"Similar-movie search failed: {error}")

    elif recommendation_mode == "Semantic Hybrid Search":
        user_id = st.number_input(
            "MovieLens User ID", min_value=1, max_value=943, value=1,
            key="movies_semantic_user",
        )
        query = st.text_area(
            "Describe the movies you want",
            placeholder="Example: I enjoy friendship, adventure and emotional journeys...",
            key="movies_semantic_query",
        )
        top_n = st.slider(
            "Number of recommendations", 5, 20, 10,
            key="movies_semantic_top_n",
        )

        if st.button("Find Movies", type="primary", width="stretch"):
            if not query.strip():
                st.warning("Please describe your movie preferences.")
            else:
                try:
                    with st.spinner("Understanding your preferences..."):
                        results = get_personalized_hybrid_recommendations(
                            int(user_id), query, top_n
                        )
                    if results.empty:
                        st.info("No matching movies found.")
                    else:
                        for _, movie in results.iterrows():
                            display_movie_card(movie)
                except Exception as error:
                    st.error(f"Semantic hybrid search failed: {error}")

    elif recommendation_mode == "Hybrid + RAG (AI Explanations)":
        st.caption(
            "Combines recommendation signals with retrieved movie metadata "
            "and AI-generated explanations."
        )
        user_id = st.number_input(
            "MovieLens User ID", min_value=1, max_value=943, value=1,
            key="movies_rag_user",
        )
        query = st.text_area(
            "What kind of movies do you enjoy?",
            placeholder=(
                "Example: I love adventure, friendship, emotional stories "
                "and inspiring journeys."
            ),
            height=120,
            key="movies_rag_query",
        )
        top_n = st.slider(
            "Number of recommendations", 3, 10, 5,
            key="movies_rag_top_n",
        )

        if st.button(
            "Generate AI Recommendations", type="primary", width="stretch"
        ):
            if not query.strip():
                st.warning("Please describe your movie preferences.")
            else:
                try:
                    with st.spinner(
                        "Finding movies and generating AI explanations..."
                    ):
                        results = generate_rag_recommendations(
                            int(user_id), query, top_n
                        )
                    if results.empty:
                        st.warning("No recommendations found.")
                    else:
                        st.success(
                            f"Found {len(results)} personalized recommendations!"
                        )
                        for _, movie in results.iterrows():
                            display_movie_card(
                                movie, explanation=movie.get("explanation")
                            )
                except Exception as error:
                    st.error(f"AI recommendations failed: {error}")

    elif recommendation_mode == "LightFM (WARP)":
        st.info(
            "LightFM uses learned user and movie-feature embeddings to rank "
            "recommendations. It does not use a text preference query."
        )
        user_id = st.number_input(
            "MovieLens User ID", min_value=1, max_value=943, value=1,
            key="movies_lightfm_user",
        )
        top_n = st.slider(
            "Number of recommendations", 3, 10, 5,
            key="movies_lightfm_top_n",
        )

        if st.button(
            "Generate LightFM Recommendations", type="primary", width="stretch"
        ):
            try:
                with st.spinner("LightFM is ranking movies for this user..."):
                    results = get_lightfm_recommendations(
                        int(user_id), top_n=top_n
                    )
                if results.empty:
                    st.warning("No LightFM recommendations found.")
                else:
                    st.success(
                        f"Generated {len(results)} LightFM recommendations!"
                    )
                    for _, movie in results.iterrows():
                        display_movie_card(movie)
            except Exception as error:
                st.error(f"LightFM recommendation failed: {error}")


# ==========================================
# BOOK RECOMMENDATIONS
# ==========================================

elif page == "Books":

    st.title("📚 Book Discovery & Recommendations")
    st.write(
        "Search books by meaning, discover similar titles, or get "
        "personalized recommendations using your Book-Crossing history."
    )

    try:
        engines = load_book_engines()
        book_recommender = engines["recommender"]
        book_semantic = engines["semantic"]
        book_hybrid = engines["hybrid"]

        mode = st.selectbox(
            "Choose book recommendation method",
            [
                "Semantic Search",
                "Collaborative Recommendations",
                "Similar Books",
                "Hybrid Recommendations",
            ],
            key="book_mode",
        )

        top_n = st.slider(
            "Number of books",
            min_value=5,
            max_value=20,
            value=10,
            key="book_top_n",
        )

        st.divider()

        if mode == "Semantic Search":
            query = st.text_area(
                "Describe the book you are looking for",
                placeholder=(
                    "Example: a thoughtful science-fiction novel about "
                    "artificial intelligence, identity and the future."
                ),
                key="book_semantic_query",
            )

            if st.button("Search Books", type="primary", width="stretch"):
                if not query.strip():
                    st.warning("Enter a description or topic to search.")
                else:
                    try:
                        with st.spinner("Finding books with semantic search..."):
                            results = book_semantic.semantic_book_search(
                                query.strip(), top_n=top_n
                            )
                        if results.empty:
                            st.info("No matching books were found.")
                        else:
                            st.success(f"Found {len(results)} books.")
                            for _, book in results.iterrows():
                                display_book_card(book)
                    except Exception as error:
                        st.error(f"Book semantic search failed: {error}")

        elif mode == "Collaborative Recommendations":
            st.caption(
                "Recommendations are based on a Book-Crossing user’s previous ratings."
            )
            user_ids = book_recommender.rating_matrix.index
            if len(user_ids) == 0:
                st.warning("No Book-Crossing users are available.")
            else:
                user_id = st.selectbox(
                    "Book-Crossing User ID",
                    options=[int(value) for value in user_ids],
                    key="book_cf_user",
                )

                if st.button(
                    "Recommend Books", type="primary", width="stretch"
                ):
                    try:
                        with st.spinner("Generating collaborative recommendations..."):
                            results = book_recommender.get_collaborative_book_recommendations(
                                int(user_id), top_n=top_n
                            )
                        if results.empty:
                            st.info("No unseen book recommendations found for this user.")
                        else:
                            st.success(f"Generated {len(results)} recommendations.")
                            for _, book in results.iterrows():
                                display_book_card(book)
                    except Exception as error:
                        st.error(f"Collaborative recommendations failed: {error}")

        elif mode == "Similar Books":
            titles = (
                book_semantic.books["Book-Title"]
                .dropna()
                .astype(str)
                .str.strip()
            )
            title_options = sorted(
                title for title in titles.unique().tolist() if title
            )

            selected_title = st.selectbox(
                "Select a book",
                options=title_options,
                key="book_similar_title",
            )

            if st.button(
                "Find Similar Books", type="primary", width="stretch"
            ):
                try:
                    with st.spinner("Finding semantically similar books..."):
                        results = book_semantic.get_semantically_similar_books(
                            selected_title, top_n=top_n
                        )
                    if results.empty:
                        st.info("No similar books were found.")
                    else:
                        st.success(f"Found {len(results)} similar books.")
                        for _, book in results.iterrows():
                            display_book_card(book)
                except Exception as error:
                    st.error(f"Similar-book search failed: {error}")

        elif mode == "Hybrid Recommendations":
            st.caption(
                "Combines collaborative filtering with semantic similarity "
                "around a selected book."
            )

            user_ids = book_recommender.rating_matrix.index
            user_id = st.selectbox(
                "Book-Crossing User ID",
                options=[int(value) for value in user_ids],
                key="book_hybrid_user",
            )

            titles = (
                book_semantic.books["Book-Title"]
                .dropna()
                .astype(str)
                .str.strip()
            )
            title_options = sorted(
                title for title in titles.unique().tolist() if title
            )
            selected_title = st.selectbox(
                "Choose a book matching your interests",
                options=title_options,
                key="book_hybrid_title",
            )

            if st.button(
                "Generate Hybrid Recommendations",
                type="primary",
                width="stretch",
            ):
                try:
                    with st.spinner("Combining collaborative and semantic signals..."):
                        results = book_hybrid.hybrid_book_recommendations(
                            user_id=int(user_id),
                            book_title=selected_title,
                            top_n=top_n,
                        )
                    if results.empty:
                        st.info("No hybrid recommendations were found.")
                    else:
                        st.success(f"Generated {len(results)} hybrid recommendations.")
                        for _, book in results.iterrows():
                            display_book_card(book)
                except Exception as error:
                    st.error(f"Hybrid recommendations failed: {error}")

    except Exception as error:
        st.error(f"Could not load the book recommendation engines: {error}")
        st.info(
            "Check that datasets/books/clean_books.csv, "
            "datasets/books/clean_ratings.csv, "
            "datasets/books/unified_book_catalog.csv, and "
            "models/book_embeddings.npy are present."
        )




# ==========================================
# PRODUCT DISCOVERY & RECOMMENDATIONS
# ==========================================

elif page == "Products":

    st.title("🛍️ Product Discovery & Recommendations")
    st.write(
        "Explore electronics through semantic search, similar-product discovery, "
        "collaborative filtering and hybrid recommendations."
    )

    try:
        engines = load_product_engines()
        product_recommender = engines["recommender"]
        product_semantic = engines["semantic"]
        product_hybrid = engines["hybrid"]
        product_catalog = engines["catalog"]

        mode = st.selectbox(
            "Choose product recommendation method",
            [
                "Semantic Product Search",
                "Collaborative Recommendations",
                "Similar Products",
                "Hybrid Recommendations",
            ],
            key="product_mode",
        )

        top_n = st.slider(
            "Number of products",
            min_value=5,
            max_value=20,
            value=10,
            key="product_top_n",
        )

        st.divider()

        if mode == "Semantic Product Search":
            query = st.text_area(
                "Describe the product you are looking for",
                placeholder=(
                    "Example: a reliable 27-inch monitor for programming, "
                    "with good colour accuracy and adjustable height."
                ),
                key="product_semantic_query",
            )

            if st.button("Search Products", type="primary", width="stretch"):
                if not query.strip():
                    st.warning("Enter a product description or search query.")
                else:
                    try:
                        with st.spinner("Searching products by meaning..."):
                            results = product_semantic.semantic_product_search(
                                query.strip(), top_n=top_n
                            )
                        display_product_results(results, product_catalog)
                    except Exception as error:
                        st.error(f"Product semantic search failed: {error}")

        elif mode == "Collaborative Recommendations":
            st.caption(
                "Enter an Amazon reviewer ID that exists in the product ratings dataset."
            )
            user_ids = product_recommender.user_product_matrix.index
            default_user = str(user_ids[0]) if len(user_ids) else ""

            user_id = st.text_input(
                "Amazon Reviewer ID",
                value=default_user,
                key="product_cf_user",
            )

            if st.button("Recommend Products", type="primary", width="stretch"):
                if not user_id.strip():
                    st.warning("Enter a reviewer ID.")
                elif user_id not in product_recommender.user_product_matrix.index:
                    st.error(
                        "That reviewer ID is not present in the product dataset. "
                        "Use an ID from the dataset."
                    )
                else:
                    try:
                        with st.spinner("Generating collaborative recommendations..."):
                            results = product_recommender.collaborative_product_recommendations(
                                user_id, top_n=top_n
                            )
                        display_product_results(results, product_catalog)
                    except Exception as error:
                        st.error(f"Collaborative recommendations failed: {error}")

        elif mode == "Similar Products":
            product_titles = (
                product_semantic.products["title"]
                .dropna()
                .astype(str)
                .str.strip()
            )
            title_options = sorted(
                title for title in product_titles.unique().tolist() if title
            )

            selected_title = st.selectbox(
                "Select a product",
                options=title_options,
                key="product_similar_title",
            )

            if st.button("Find Similar Products", type="primary", width="stretch"):
                try:
                    with st.spinner("Finding similar products..."):
                        results = product_semantic.get_similar_products(
                            selected_title, top_n=top_n
                        )
                    display_product_results(results, product_catalog)
                except Exception as error:
                    st.error(f"Similar-product search failed: {error}")

        elif mode == "Hybrid Recommendations":
            st.caption(
                "Combines collaborative signals from a reviewer’s history "
                "with semantic matching to your product description."
            )

            user_ids = product_recommender.user_product_matrix.index
            default_user = str(user_ids[0]) if len(user_ids) else ""

            user_id = st.text_input(
                "Amazon Reviewer ID",
                value=default_user,
                key="product_hybrid_user",
            )

            query = st.text_area(
                "Describe what kind of product you want",
                placeholder=(
                    "Example: wireless noise-cancelling headphones suitable "
                    "for studying and long work sessions."
                ),
                key="product_hybrid_query",
            )

            if st.button(
                "Generate Hybrid Recommendations",
                type="primary",
                width="stretch",
            ):
                if not user_id.strip():
                    st.warning("Enter a reviewer ID.")
                elif user_id not in product_recommender.user_product_matrix.index:
                    st.error("That reviewer ID is not present in the product dataset.")
                elif not query.strip():
                    st.warning("Describe the type of product you want.")
                else:
                    try:
                        with st.spinner("Combining your history and product preferences..."):
                            results = product_hybrid.hybrid_product_recommendations(
                                user_id=user_id,
                                query=query.strip(),
                                top_n=top_n,
                            )
                        display_product_results(results, product_catalog)
                    except Exception as error:
                        st.error(f"Hybrid recommendations failed: {error}")

    except Exception as error:
        st.error(f"Could not load the product recommendation engines: {error}")
        st.info(
            "Check that datasets/products/processed/electronics_reviews.csv, "
            "datasets/products/processed/electronics_catalog.csv, "
            "models/semantic_product_features.csv, and "
            "models/product_embeddings.npy are present."
        )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "RecomMind AI | Movies + Books + Products | Hybrid AI Recommendation Platform"
)

