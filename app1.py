import streamlit as st
import pickle
import pandas as pd
import requests
import os
from dotenv import load_dotenv

# -----------------------------
# Load .env file for API key
# -----------------------------
load_dotenv()
OMDB_API_KEY = st.secrets["OMDB_API_KEY"]

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="centered"
)

# -----------------------------
# Sidebar Info
# -----------------------------
st.sidebar.title("ℹ️ Project Details")
st.sidebar.markdown("### 📂 Dataset Name")
st.sidebar.write("TMDB 5000 Movies Dataset")
st.sidebar.markdown("### 🧠 Model")
st.sidebar.write("Count Vectorizer + Cosine Similarity")
st.sidebar.markdown("### ⚙️ Type")
st.sidebar.write("Content-Based Recommendation System")
st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 Built with **Streamlit**")

# -----------------------------
# Dark Mode Styling (Optional)
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #fafafa;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Pickled Files
# -----------------------------
movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

if not isinstance(movies, pd.DataFrame):
    movies = pd.DataFrame(movies)

# -----------------------------
# OMDb Poster Function
# -----------------------------
def fetch_poster(title):
    url = "http://www.omdbapi.com/"
    params = {
        "apikey": OMDB_API_KEY,
        "t": title
    }
    response = requests.get(url, params=params).json()
    poster_url = response.get("Poster")
    if poster_url and poster_url != "N/A":
        return poster_url
    return "https://via.placeholder.com/300x450?text=No+Poster"

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_names = []
    recommended_posters = []

    for i in movies_list:
        title = movies.iloc[i[0]].title
        recommended_names.append(title)
        recommended_posters.append(fetch_poster(title))

    return recommended_names, recommended_posters

# -----------------------------
# Main App
# -----------------------------
st.title("🎬 Movie Recommendation System")
st.write("Select a movie to get similar recommendations")

selected_movie = st.selectbox(
    "🎥 Choose a Movie",
    movies['title'].values
)

if st.button("🔍 Recommend"):
    names, posters = recommend(selected_movie)

    st.subheader("✨ Recommended Movies")
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.image(posters[i])
            st.caption(names[i])