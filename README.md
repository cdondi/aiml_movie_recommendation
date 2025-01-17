
# **Movie Recommendation System**

A hybrid movie recommendation system built using **content-based filtering** and **collaborative filtering**, deployed as an interactive web application with **Streamlit**.

## **Features**

- **Content-Based Recommendations**: Suggests movies similar to a given movie title based on metadata (e.g., genres, title).
- **Collaborative Filtering**: Suggests movies based on user preferences and interactions (e.g., ratings).
- **Hybrid Recommendations**: Combines content-based and collaborative filtering with adjustable weights.
- User-friendly interface with options to adjust parameters and get personalized recommendations.

---

## **Table of Contents**

1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Usage](#usage)
4. [Key Features](#key-features)
5. [Technologies Used](#technologies-used)
6. [License](#license)

---

## **Installation**

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Data**:
   - Place your dataset files (`movies.csv`, `ratings.csv`) in the `data/` folder.
   - Ensure the required models (`cosine_sim.h5`, `user_factors.pkl`, `movie_factors.pkl`, `movie_indices.pkl`) are in the `models/` folder.

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

5. Open your browser and navigate to `http://localhost:8501` (or the URL shown in your terminal).

---

## **Project Structure**

```
movie-recommendation-system/
│
├── app.py                   # Main Streamlit app
├── utils/
│   ├── recommendation.py    # Recommendation logic (content-based, collaborative, hybrid)
│   ├── data_loader.py        # Data loading functions (e.g., get_movies())
│
├── models/
│   ├── cosine_sim.h5        # Precomputed cosine similarity matrix for content-based filtering
│   ├── user_factors.pkl     # User latent factors for collaborative filtering
│   ├── movie_factors.pkl    # Movie latent factors for collaborative filtering
│   ├── movie_indices.pkl    # Mapping of movie titles to indices
│
├── data/
│   ├── movies.csv           # Movies metadata
│   ├── ratings.csv          # User ratings
│
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## **Usage**

1. **Enter a Movie Title**:
   - Input a movie title (e.g., `"Toy Story (1995)"`) to get recommendations.

2. **Optionally Enter a User ID**:
   - Provide a user ID to personalize recommendations using collaborative filtering.

3. **Adjust Weights**:
   - Use the sliders to control the balance between content-based and collaborative recommendations.

4. **Set Number of Recommendations**:
   - Choose how many recommendations to display.

5. **View Results**:
   - The app displays a list of recommended movies based on the inputs and settings.

---

## **Key Features**

- **Customizable Hybrid Recommendations**:
   - Adjust weights for content-based and collaborative filtering to get tailored results.

- **Efficient Memory Usage**:
   - Precomputed cosine similarity matrix stored in HDF5 format for content-based filtering.
   - Optimized SVD for collaborative filtering.

- **Interactive Interface**:
   - Built with Streamlit for a user-friendly experience.

---

## **Technologies Used**

- **Python**: Core programming language.
- **Streamlit**: For building the web interface.
- **pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computations.
- **scikit-learn**: TF-IDF vectorization and Truncated SVD.
- **h5py**: Handling HDF5 files for cosine similarity.
- **Matplotlib**: (Optional) For data visualization.

---

## **License**

This project is licensed under the [MIT License](LICENSE).
