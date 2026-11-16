import streamlit as st
import pandas as pd

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
    get_engine_info
)

from modules.rag_pipeline import generate_rag_recommendations


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="RecomMind AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


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

</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🎬 RecomMind AI")

    st.caption("Intelligent Recommendation Platform")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Discover Movies",
            "Personalized Recommendations"
        ]
    )

    st.divider()

    st.caption("Powered by Hybrid AI Recommendation")


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


# ==========================================
# DISCOVER MOVIES
# ==========================================

elif page == "Discover Movies":

    st.title("🔍 Discover Movies")

    mode = st.selectbox(
        "Choose recommendation method",
        [
            "Collaborative Filtering",
            "Similar Movies",
            "Semantic Hybrid Search"
        ]
    )

    st.divider()

    if mode == "Collaborative Filtering":

        user_id = st.number_input(
            "Enter MovieLens User ID",
            min_value=1,
            max_value=943,
            value=1
        )

        top_n = st.slider(
            "Number of recommendations",
            5, 20, 10
        )

        if st.button("Recommend Movies", type="primary"):

            try:

                with st.spinner("Finding recommendations..."):

                    results = get_collaborative_recommendations(
                        int(user_id),
                        top_n
                    )

                st.success("Recommendations generated!")

                for _, movie in results.iterrows():

                    st.markdown(
                        f"""
                        <div class="movie-card">
                            <div class="movie-title">
                                {movie['title']}
                            </div>
                            <div class="movie-meta">
                                Genres: {movie['genre_text']}
                            </div>
                            <div class="movie-meta">
                                Predicted rating:
                                {movie['predicted_rating']:.2f}/5
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            except Exception as error:
                st.error(str(error))


    elif mode == "Similar Movies":

        movie_options = sorted(
            pd.Series(
                __import__(
                    "modules.movie_recommender",
                    fromlist=["movie_features"]
                ).movie_features["title"]
            ).dropna().unique().tolist()
        )

        selected_movie = st.selectbox(
            "Select a movie",
            movie_options
        )

        top_n = st.slider(
            "Number of similar movies",
            5, 20, 10
        )

        if st.button("Find Similar Movies", type="primary"):

            try:

                results = get_content_recommendations(
                    selected_movie,
                    top_n
                )

                for _, movie in results.iterrows():

                    st.markdown(
                        f"""
                        <div class="movie-card">
                            <div class="movie-title">
                                {movie['title']}
                            </div>
                            <div class="movie-meta">
                                Genres: {movie['genre_text']}
                            </div>
                            <div class="movie-meta">
                                Similarity:
                                {movie['content_similarity']:.3f}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            except Exception as error:
                st.error(str(error))


    elif mode == "Semantic Hybrid Search":

        user_id = st.number_input(
            "MovieLens User ID",
            min_value=1,
            max_value=943,
            value=1,
            key="semantic_user"
        )

        query = st.text_area(
            "Describe the movies you want",
            placeholder="Example: I enjoy friendship, adventure and emotional journeys..."
        )

        top_n = st.slider(
            "Number of recommendations",
            5, 20, 10,
            key="semantic_top_n"
        )

        if st.button("Find Movies", type="primary"):

            try:

                with st.spinner("Understanding your preferences..."):

                    results = get_personalized_hybrid_recommendations(
                        int(user_id),
                        query,
                        top_n
                    )

                for _, movie in results.iterrows():

                    st.markdown(
                        f"""
                        <div class="movie-card">
                            <div class="movie-title">
                                {movie['title']}
                            </div>
                            <div class="movie-meta">
                                {movie['genre_text']}
                            </div>
                            <div class="movie-meta">
                                Semantic similarity:
                                {movie['semantic_similarity']:.3f}
                            </div>
                            <div class="movie-meta">
                                Hybrid score:
                                {movie['hybrid_score']:.3f}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            except Exception as error:
                st.error(str(error))


# ==========================================
# PERSONALIZED AI RECOMMENDATIONS
# ==========================================

elif page == "Personalized Recommendations":

    st.title("✨ AI Personalized Recommendations")

    st.write(
        "Describe your movie preferences and let RecomMind AI "
        "find movies suited to your interests."
    )

    user_id = st.number_input(
        "MovieLens User ID",
        min_value=1,
        max_value=943,
        value=1,
        key="rag_user"
    )

    query = st.text_area(
        "What kind of movies do you enjoy?",
        placeholder=(
            "Example: I love adventure, friendship, "
            "emotional stories and inspiring journeys."
        ),
        height=120
    )

    top_n = st.slider(
        "Number of recommendations",
        3, 10, 5
    )

    if st.button(
        "Generate AI Recommendations",
        type="primary",
        use_container_width=True
    ):

        if not query.strip():

            st.warning("Please describe your movie preferences.")

        else:

            try:

                with st.spinner(
                    "RecomMind AI is finding your movies and "
                    "generating explanations..."
                ):

                    results = generate_rag_recommendations(
                        int(user_id),
                        query,
                        top_n
                    )

                if results.empty:

                    st.warning("No recommendations found.")

                else:

                    st.success(
                        f"Found {len(results)} personalized recommendations!"
                    )

                    for _, movie in results.iterrows():

                        st.markdown(
                            f"""
                            <div class="movie-card">

                                <div class="movie-title">
                                    🎬 {movie['title']}
                                </div>

                                <div class="movie-meta">
                                    Genres: {movie['genres']}
                                </div>

                                <div class="movie-meta">
                                    ⭐ Average Rating:
                                    {movie['rating']}/5
                                </div>

                                <div class="movie-meta">
                                    Semantic Similarity:
                                    {movie['semantic_similarity']}
                                </div>

                                <div class="movie-meta">
                                    Hybrid Score:
                                    {movie['hybrid_score']}
                                </div>

                                <hr>

                                <div class="movie-explanation">
                                    <b>🤖 AI Explanation</b><br>
                                    {movie['explanation']}
                                </div>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            except Exception as error:

                st.error(f"Something went wrong: {error}")


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "RecomMind AI | Hybrid Recommendation System | "
    "MovieLens + Sentence Transformers + Qwen + LoRA"
)