import streamlit as st
import pandas as pd
import pickle
from utils.recommendation import recommend_hybrid, recommend_content_based, recommend_collaborative

# Load data and models
# @st.cache_data
# def load_data():
#     movies = pd.read_csv("data/cleaned_remapped_movies.csv")
#     with open("models/movie_indices.pkl", "rb") as f:
#         movie_indices = pickle.load(f)
#     return movies, movie_indices

# movies, movie_indices = load_data()

st.title("Movie Recommendation System 🎥🍿")

# Sidebar inputs
st.sidebar.header("Inputs")
title = st.sidebar.text_input("Enter a Movie Title", placeholder="e.g., Toy Story (1995)")
# user_id = st.sidebar.number_input("Enter User ID (Optional)", min_value=1, step=1, value=1)
# 200948
user_id = st.sidebar.selectbox(
    "Enter User ID (Optional)", 
    options=[None] + list(range(1, 101)),  # Replace 101 with your maximum user ID
    format_func=lambda x: "None" if x is None else f"User {x}"
)
content_weight = st.sidebar.slider("Content Weight", 0.0, 1.0, 0.5)
collab_weight = 1.0 - content_weight
num_recommendations = st.sidebar.slider("Number of Recommendations", 1, 20, 10)

# Recommendation logic
if st.sidebar.button("Get Recommendations"):
    try:
        # Hybrid recommendation
        recommendations = recommend_hybrid(
            title=title,
            user_id=user_id if user_id else None,
            content_weight=content_weight,
            collab_weight=collab_weight,
            num_recommendations=num_recommendations
        )
        st.subheader("Recommended Movies:")
        st.write(recommendations)
    except Exception as e:
        st.error(f"Error: {e}")