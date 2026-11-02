import streamlit as st
import pandas as pd

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


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #0e1117;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #ffffff;
}

.subtitle {
    color: #a0a0a0;
    font-size: 17px;
}

.movie-card {
    background-color: #1c2230;
    padding: 20px;
    border-radius: 14px;
    margin-bottom: 15px;
    border: 1px solid #303848;
}

.movie-title {
    font-size: 21px;
    font-weight: bold;
    color: #ffffff;
}

.movie-meta {
    color: #b0b8c5;
    font-size: 14px;
}

.movie-explanation {
    color: #e0e0e0;
    line-height: 1.6;
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