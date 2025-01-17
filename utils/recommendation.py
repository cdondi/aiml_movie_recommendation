import pandas as pd
import numpy as np
import pickle
# from scipy.sparse import csr_matrix
# from sklearn.metrics.pairwise import cosine_similarity
import h5py
import streamlit as st

_movies = None

def get_movies():
    """
    Load the movies dataset as a singleton.
    Returns:
        pd.DataFrame: Movies DataFrame.
    """
    global _movies
    if _movies is None:
        _movies = pd.read_csv("data/cleaned_remapped_movies.csv")
    return _movies

# Load models and data
def load_models():
    with open("models/movie_indices.pkl", "rb") as f:
        movie_indices = pickle.load(f)
    with open("models/user_factors.pkl", "rb") as f:
        user_factors = pickle.load(f)
    with open("models/movie_factors.pkl", "rb") as f:
        movie_factors = pickle.load(f)
    return movie_indices, user_factors, movie_factors


movie_indices, user_factors, movie_factors = load_models()


# Content-Based Recommendation
def recommend_content_based(title, num_recommendations=10, h5_file='models/cosine_sim.h5', min_similarity=0.4):
    movies = get_movies()
    # st.write(":::::::: Content-Based Recommendations .....")
    """
    Recommend movies similar to the given title using content-based filtering.
    Adding a min_similarity threshhold dramatically improves this content-based recommendation especially when used
    along in a hybrid recommendation system

    Args:
        title (str): Movie title.
        num_recommendations (int): Number of recommendations.

    Returns:
        list: Recommended movie titles.
    """
    # Normalize the movie title
    normalized_title = title.lower().strip()
    
    # Get the index of the input movie
    if normalized_title not in movie_indices:
        raise ValueError(f"Movie '{title}' not found in the dataset.")
    idx = movie_indices[normalized_title]
    
    # Open the HDF5 file and retrieve the relevant row (cosine similarity scores)
    # Instead of loading the entire cosine_sim matrix into memory, retrieve only the relevant row using the index idx.
    with h5py.File(h5_file, 'r') as f:
        sim_scores = f['cosine_sim'][idx]  # Retrieve the row corresponding to the movie

    # Process the similarity scores to get recommendations
    sim_scores = list(enumerate(sim_scores))

    # Filter by similarity threshold
    sim_scores = [(i, score) for i, score in sim_scores if score >= min_similarity]
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]  # Skip the movie itself
    recommended_indices = [i[0] for i in sim_scores]
    
    # Retrieve recommended movie titles and convert to title case
    recommended_titles = movies['title'].iloc[recommended_indices].str.title()
    return recommended_titles


# Collaborative Filtering Recommendation
def recommend_collaborative(user_id, num_recommendations=5):
    movies = get_movies()
    # st.write(":::::::: Collaborative Recommendations .........")
    """
    Recommend movies for a user using collaborative filtering.

    Args:
        user_id (int): User ID.
        num_recommendations (int): Number of recommendations.

    Returns:
        list: Recommended movie titles.
    """
    # # Load user and movie factors
    # with open("../models/user_factors.pkl", "rb") as f:
    #     user_factors = pickle.load(f)
    # with open("../models/movie_factors.pkl", "rb") as f:
    #     movie_factors = pickle.load(f)

    # Validate user_id
    if user_id is None:
        raise ValueError("Invalid user_id: user_id cannot be empty")
    if user_id < 1 or user_id > user_factors.shape[0] or user_id is None:
        raise ValueError(f"Invalid user_id: {user_id}. Must be between 1 and {user_factors.shape[0]}.")

    # Retrieve the user's latent factor vector
    user_vector = user_factors[user_id - 1]

    # Compute similarity scores for all movies
    scores = np.dot(movie_factors, user_vector)

    # Get indices of the top movie scores
    recommended_movie_indices = np.argsort(scores)[::-1][:num_recommendations]

    # Map indices to movie titles
    recommended_titles = movies.loc[movies.index.isin(recommended_movie_indices), 'title'].tolist()
    return recommended_titles


# Hybrid Recommendation
def recommend_hybrid(title, user_id=None, content_weight=0.5, collab_weight=0.5, num_recommendations=10):
    # st.write(":::::::: Hybrid Recommendation ......:")
    """
    Hybrid recommendation combining content-based and collaborative filtering.
    
    Args:
        title (str): Input movie title for content-based recommendations.
        user_id (int): User ID for collaborative filtering recommendations.
        content_weight (float): Weight for content-based recommendations.
        collab_weight (float): Weight for collaborative filtering recommendations.
        num_recommendations (int): Number of recommendations to return.
    
    Returns:
        list: Recommended movie titles.
    """
    # st.write(":::::::: Collab weight => ", collab_weight)
    # st.write(":::::::: Content weight => ", content_weight)
    # st.write(":::::::: User ID => ", user_id)
    # Validate weights
    if content_weight + collab_weight != 1.0:
        raise ValueError("Content weight and collaboration weight must sum to 1.0.")
        
    # Content-based recommendations
    content_recommendations = recommend_content_based(title, num_recommendations=100)

    # Collaborative recommendations
    if user_id is None:
        collaborative_recommendations = []
    else:
        collaborative_recommendations = recommend_collaborative(user_id, num_recommendations=100)

    # Combine scores (assume both return lists of movie titles)
    combined_recommendations = (
        content_weight * pd.Series(content_recommendations).value_counts(normalize=True) +
        collab_weight * pd.Series(collaborative_recommendations).value_counts(normalize=True)
    ).sort_values(ascending=False)

    # # Return top-N recommendations
    # return combined_recommendations.head(num_recommendations).index.tolist()
    # Convert to title case
    top_recommendations = combined_recommendations.head(num_recommendations).index.tolist()
    top_recommendations_title_case = [title.title() for title in top_recommendations]

    # Return top-N recommendations in title case
    return top_recommendations_title_case