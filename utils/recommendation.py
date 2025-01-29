import boto3
import os
import io
import faiss
import pandas as pd
import numpy as np
import pickle
import h5py
import streamlit as st
import fsspec
from sklearn.feature_extraction.text import TfidfVectorizer

_movies = None
_movie_vectors = None
# AWS S3 Configuration
BUCKET_NAME = "movielens-clived"  # Change to your bucket name

@st.cache_resource
def download_model(filename, bucket_name=BUCKET_NAME):
    """
    Download the model file from AWS S3 if not already downloaded.

    Args:
        filename (str): The name of the file to download.
        bucket_name (str): The AWS S3 bucket containing the model.

    Returns:
        str: The local path of the downloaded model file.
    """
    local_model_path = f"../models/{filename}"

    # Check if the model already exists locally
    if not os.path.exists(local_model_path):
        st.write(f":::::::::{local_model_path}")
        st.write(f"📥 Downloading {filename} from S3...")

        # Load AWS credentials (Streamlit secrets first, fallback to environment variables)
        aws_access_key = st.secrets["MOVIELENS_AWS_ACCESS_KEY_ID"] if "MOVIELENS_AWS_ACCESS_KEY_ID" in st.secrets else os.getenv("MOVIELENS_AWS_ACCESS_KEY_ID")
        aws_secret_key = st.secrets["MOVIELENS_AWS_SECRET_ACCESS_KEY"] if "MOVIELENS_AWS_SECRET_ACCESS_KEY" in st.secrets else os.getenv("MOVIELENS_AWS_SECRET_ACCESS_KEY")

        # Initialize S3 client
        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )

        try:
            # Download model file
            with open(local_model_path, "wb") as f:
                s3.download_fileobj(bucket_name, filename, f)
            st.write(f"✅ Model downloaded and saved as {local_model_path}")
        except Exception as e:
            st.error(f"❌ Failed to download {filename} from S3: {e}")
            return None

    return local_model_path

@st.cache_resource
def load_models():
    """Load the models from disk."""
    model_path = download_model('movie_factors.pkl')
    with open(model_path, "rb") as f:
        movie_factors = pickle.load(f)

    model_path = download_model('movie_indices.pkl')
    with open(model_path, "rb") as f:
        movie_indices = pickle.load(f)

    model_path = download_model('user_factors.pkl')
    with open(model_path, "rb") as f:
        user_factors = pickle.load(f)

    # model_path = download_model('faiss_gpu_index40k-1024.bin')
    model_path = download_model('faiss_gpu_index60k-1800.bin')
    with open(model_path, "rb") as f:
        faiss_index = faiss.read_index(model_path)

    return movie_indices, user_factors, movie_factors, faiss_index

def load_csv_from_s3(filename):
    """Fetch CSV file from S3 and load it into a Pandas DataFrame."""
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("MOVIELENS_AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("MOVIELENS_AWS_SECRET_ACCESS_KEY")
    )

    response = s3.get_object(Bucket=BUCKET_NAME, Key=filename)
    return pd.read_csv(io.BytesIO(response["Body"].read()))

def get_movies():
    """
    Load the movies dataset as a singleton.
    Returns:
        pd.DataFrame: Movies DataFrame.
    """
    global _movies
    if _movies is None:
        # _movies = pd.read_csv("data/cleaned_remapped_movies.csv")
        _movies = load_csv_from_s3('cleaned_remapped_movies.csv')

    return _movies

def get_movie_vectors():
    """
    Load the TF-IDF matrix as a singleton.
    Returns:
        csr_matrix: TF-IDF matrix.
    """

    global _movie_vectors
    if _movie_vectors is None:
        # Initialize TF-IDF Vectorizer
        st.write("___________ Initializing TFIDF Matrix .................")
        tfidf = TfidfVectorizer(stop_words='english')

        # Fit and transform the combined features
        tfidf_matrix = tfidf.fit_transform(_movies['combined_features'])
        st.write("___________ TFIDF Matrix initialized .................")

        # Convert the sparse TF-IDF matrix to a dense NumPy array for FAISS
        st.write("___________ Convert sparse TFIDF Matrix to dense numpy array for FAISS  ..........")
        _movie_vectors = tfidf_matrix.toarray().astype('float32')
        st.write("___________ Dense numpy array for FAISS initialized  ..........")

    return _movie_vectors

st.write("::::::::::: Loading models .................")
movie_indices, user_factors, movie_factors, faiss_index = load_models()
st.write("::::::::::: Models loaded .................")

def recommend_content_based(title, num_recommendations=10, min_similarity=0.4):
    """
    Recommend movies similar to the given title using FAISS for content-based filtering.

    Args:
        title (str): Movie title.
        num_recommendations (int): Number of recommendations.
        min_similarity (float): Minimum similarity threshold.

    Returns:
        list: Recommended movie titles.
    """
    # Normalize the movie title
    normalized_title = title.lower().strip()

    # Get the index of the input movie
    if normalized_title not in movie_indices:
        raise ValueError(f"Movie '{title}' not found in the dataset.")
    
    idx = movie_indices[normalized_title]

    movies = get_movies()
    
    st.write("::::::::::: Initializing movie vectors .................")
    movie_vectors = get_movie_vectors()
    

    # # Convert the sparse TF-IDF matrix to a dense NumPy array for FAISS
    # movie_vectors = tfidf_matrix.toarray().astype('float32')

    # Retrieve similarity scores from FAISS
    st.write("::::::::::: Searching FAISS for Similar Movies .................")
    query_vector = movie_vectors[idx].reshape(1, -1).astype('float32')
    distances, recommended_indices = faiss_index.search(query_vector, num_recommendations + 1)  # +1 to skip the input movie
    st.write("::::::::::: FAISS Search Complete .................")

    # Process similarity scores
    sim_scores = list(zip(recommended_indices[0], distances[0]))

    # Filter by similarity threshold
    sim_scores = [(i, score) for i, score in sim_scores if score >= min_similarity]

    st.write("::::::::::: Applying Similarity Threshold .................")

    # Sort by similarity (Descending)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]  # Skip the movie itself
    recommended_indices = [i[0] for i in sim_scores]

    st.write("::::::::::: Mapping movie titles to recommended indices .................")

    # Retrieve recommended movie titles and convert to title case
    recommended_titles = movies['title'].iloc[recommended_indices].str.title()

    return recommended_titles


# Collaborative Filtering Recommendation
def recommend_collaborative(user_id, num_recommendations=5):
    """
    Recommend movies for a user using collaborative filtering.

    Args:
        user_id (int): User ID.
        num_recommendations (int): Number of recommendations.

    Returns:
        list: Recommended movie titles.
    """

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
    movies = get_movies()
    recommended_titles = movies.loc[movies.index.isin(recommended_movie_indices), 'title'].tolist()
    return recommended_titles


# Hybrid Recommendation
def recommend_hybrid(title, user_id=None, content_weight=0.5, collab_weight=0.5, num_recommendations=10):
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