import streamlit as st
import pickle
import pandas as pd
import requests

movies = pickle.load(open('movies.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("Movie Recommender System")
st.subheader("Find movies similar to your favorites using Machine Learning")

def fetch_poster(movie_id):

    api_key = st.secrets["TMDB_API_KEY"]
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    
    data = requests.get(url)
    data = data.json()

    poster_path = data['poster_path']

    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path

    return full_path

def recommend(movie):

    movie_index = movies[movies['title'] == movie].index[0]

    distances = similarity[movie_index]

    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_posters


st.title('Movie Recommender System')

selected_movie = st.selectbox(
    "Search a movie",
    movies['title'].sort_values().values
)

if st.button('Recommend'):
    with st.spinner("Finding similar movies..."):
    
        names, posters = recommend(selected_movie)

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image(posters[0], use_container_width=True)
            st.caption(names[0])

        with col2:
            st.image(posters[1], use_container_width=True)
            st.caption(names[1])

        with col3:
            st.image(posters[2], use_container_width=True)
            st.caption(names[2])

        with col4:
            st.image(posters[3], use_container_width=True)
            st.caption(names[3])

        with col5:
            st.image(posters[4], use_container_width=True)
            st.caption(names[4])
