
# Project Roadmap: FAISS-Based Movie Recommendation System

## Project Overview
This project involved building a scalable, efficient movie recommendation system using:
- Collaborative Filtering (with FAISS)
- Content-Based Filtering (TF-IDF & FAISS)
- Hybrid Recommendation combining both techniques

The main objectives were to ensure fast search, memory efficiency, and scalability as the dataset grows.

## Tools & Technologies Used

| Tool                               | Purpose                                                      |
|------------------------------------|--------------------------------------------------------------|
| FAISS                              | Fast Approximate Nearest Neighbors (ANN) for large datasets  |
| Scikit-Learn (TF-IDF Vectorizer)   | Convert movie descriptions into numerical vectors            |
| Pandas & NumPy                     | Data manipulation                                            |
| HDF5 & Pickle                      | Model storage and retrieval                                  |
| AWS S3                             | Cloud storage for models and datasets                        |
| Streamlit                          | Web UI for user interaction                                  |
| EC2 GPU Instance (g5.2xlarge)      | Efficient FAISS training                                     |

## Data Preprocessing & Model Training

| Step                                      | Description                                        | Purpose                                  |
|-------------------------------------------|----------------------------------------------------|------------------------------------------|
| Cleaned & Remapped movieId                | Ensured sequential IDs                             | Required for consistent FAISS indexing   |
| Removed Duplicate Movies                  | Merged same-title movies with different genres     | Prevents duplicate results               |
| Stored TF-IDF Matrix in csr_matrix format | Sparse format used | Reduced memory usage          |                                          |
| Trained FAISS Index (IndexIVFPQ)          | Used PQ compression and clustering                 | Enabled fast approximate search          |
| Saved Models Efficiently                  | Pickle, FAISS, and HDF5 | Enabled scalable loading |                                          |

## Challenges & Solutions

| Issue | Solution | Impact |
|-------|----------|--------|
| Large FAISS Index | Used memory-mapped FAISS (mmap=True) | Reduced RAM usage |
| FAISS Training Crashes | Batched training, used temp GPU memory | Prevented GPU memory overflow |
| CUDA Out of Memory | Used setTempMemory(1GB) and trained in iterations | Enabled successful training |
| Slow Search Performance | Tuned NUM_CLUSTERS and nprobe | Improved response speed |
| Inefficient File Loading | Implemented lazy loading and caching | Reduced redundancy |

## Model Loading & Deployment

### Optimizations Implemented:
- Memory-mapped FAISS
- Lazy-loading with `_var is None` pattern
- Singleton pattern for movie data and TF-IDF matrix
- Sparse TF-IDF matrix via `csr_matrix`
- Used AWS S3 for remote storage

## Core Recommendation Functions

| Function | Description |
|----------|-------------|
| `recommend_content_based(title, num_recommendations, min_similarity)` | Uses FAISS + TF-IDF for content similarity |
| `recommend_collaborative(user_id, num_recommendations)` | Uses user latent vectors for suggestions |
| `recommend_hybrid(title, user_id, content_weight, collab_weight)` | Blends both methods |

### Example FAISS Query Execution:
```python
query_vector = movie_vectors[idx].toarray().reshape(1, -1).astype('float32')
distances, recommended_indices = faiss_index.search(query_vector, num_recommendations + 1)
```

## Deployment

### Streamlit Integration:
- Built a lightweight frontend with Streamlit
- Used `st.cache_resource` to optimize model reloading
- Downloaded large models from AWS S3 only when needed

### UI Flow:
1. User selects a movie → `recommend_content_based()` is called
2. User inputs a user ID → `recommend_collaborative()` runs
3. Both inputs → `recommend_hybrid()` executes

## Summary of Optimizations

| Best Practice | Benefit |
|---------------|---------|
| Memory-mapped FAISS | Reduces RAM pressure |
| Sparse TF-IDF (csr_matrix) | Lower memory footprint |
| Lazy Model Loading | Avoids repeated computation |
| Streamlit Caching | Increases responsiveness |
| Batched FAISS Training | Prevents crashes |
| GPU temp memory control | Prevents CUDA errors |

## Final Achievements

- Built and deployed a scalable, hybrid recommendation engine
- Handled large data volumes with memory efficiency
- Optimized training, retrieval, and deployment pipeline
- Successfully used industry-grade tools like FAISS, AWS, and Streamlit
