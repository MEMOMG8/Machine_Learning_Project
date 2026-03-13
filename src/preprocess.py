from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import ast
import pickle


def convert(text):
    items = ast.literal_eval(text)
    return [item["name"] for item in items]


def convert_cast(text):
    items = ast.literal_eval(text)
    return [item["name"] for item in items[:3]]


def fetch_director(text):
    items = ast.literal_eval(text)
    for item in items:
        if item["job"] == "Director":
            return [item["name"]]
    return []


def clean_data(movie_list):
    return [item.replace(" ", "") for item in movie_list]


# Load datasets
movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")

# Merge datasets
movies = movies.merge(credits, on="title")

# Keep only useful columns
movies = movies[[
    "movie_id",
    "title",
    "overview",
    "genres",
    "keywords",
    "cast",
    "crew"
]]

# Remove missing values
movies.dropna(inplace=True)

# Convert JSON-like strings into lists
movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)
movies["cast"] = movies["cast"].apply(convert_cast)
movies["crew"] = movies["crew"].apply(fetch_director)

# Split overview into words
movies["overview"] = movies["overview"].apply(lambda x: x.split())

# Remove spaces inside multi-word names
movies["genres"] = movies["genres"].apply(clean_data)
movies["keywords"] = movies["keywords"].apply(clean_data)
movies["cast"] = movies["cast"].apply(clean_data)
movies["crew"] = movies["crew"].apply(clean_data)

# Create tags column
movies["tags"] = (
    movies["overview"] +
    movies["genres"] +
    movies["keywords"] +
    movies["cast"] +
    movies["crew"]
)

# Keep final columns
new_df = movies[["movie_id", "title", "tags"]].copy()

# Convert tags list into string
new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x))

# Lowercase tags
new_df["tags"] = new_df["tags"].apply(lambda x: x.lower())

print(new_df.head())
print(new_df.shape)

# Vectorize tags
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# Compute similarity matrix
similarity = cosine_similarity(vectors)

print("Similarity matrix shape:", similarity.shape)
def recommend(movie):
    
    movie_index = new_df[new_df['title'] == movie].index[0]

    distances = similarity[movie_index]

    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)

recommend("Avatar")

pickle.dump(new_df, open('movies.pkl','wb'))
pickle.dump(similarity, open('similarity.pkl','wb'))
