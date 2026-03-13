import os
import pickle
import requests
import streamlit as st

st.set_page_config(page_title="Movie Recommender", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_data
def load_data():
    with open(os.path.join(BASE_DIR, "movies.pkl"), "rb") as f:
        movies = pickle.load(f)

    with open(os.path.join(BASE_DIR, "similarity.pkl"), "rb") as f:
        similarity = pickle.load(f)

    return movies, similarity


def fetch_poster(movie_id):
    try:
        api_key = st.secrets["TMDB_API_KEY"]
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        poster_path = data.get("poster_path")
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
        return None
    except Exception:
        return None


def recommend(movie):
    movie_index = movies[movies["title"] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_posters


try:
    movies, similarity = load_data()
except Exception as e:
    st.error(f"Error loading data files: {e}")
    st.stop()


st.title("Movie Recommender System")
st.subheader("Find movies similar to your favorites using Machine Learning")

selected_movie = st.selectbox(
    "Search a movie",
    movies["title"].sort_values().values
)

if st.button("Recommend"):
    with st.spinner("Finding similar movies..."):
        names, posters = recommend(selected_movie)

    cols = st.columns(5)

    for i in range(5):
        with cols[i]:
            if posters[i]:
                st.image(posters[i], use_container_width=True)
            else:
                st.write("Poster not available")
            st.caption(names[i])