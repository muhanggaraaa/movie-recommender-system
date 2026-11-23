# Movie Recommender System - Collaborative Filtering

A small data science project that builds a movie recommendation system from the public [MovieLens 100K](https://grouplens.org/datasets/movielens/) dataset (100,836 ratings, 610 users, 9,724 movies), using item-based collaborative filtering.

## What it does

- Explores rating patterns: distribution, most-rated movies, highest-rated movies, and the sparsity of the user x movie matrix (98.3% empty).
- Builds three predictors and compares them honestly on the same test set:
  - Global mean baseline
  - Per-movie mean baseline
  - Item-based collaborative filtering (cosine similarity between movies, weighted-average prediction)
- Generates a top-10 recommendation list for a sample user.

## Results

| Method | RMSE (lower is better) |
|---|---|
| Global mean | 1.049 |
| Per-movie mean | 0.983 |
| Item-based CF | **0.861** |

## The honest part

The first version of the recommender was bad: obscure movies with almost no data scored a fake perfect 5.00 and filled the whole top-10 list. The fix was to require minimum support before trusting a prediction:

- a movie needs at least 30 ratings in the training data,
- a prediction needs at least 5 neighbor movies,
- the score is shrunk toward the movie's own average when support is weak.

After that, the recommendations actually match the user's taste (a classic-animation and adventure fan gets Howl's Moving Castle, Raiders of the Lost Ark, Hot Fuzz, and similar titles).

## How to run

```bash
# 1. download the dataset (not included in this repo)
curl -O https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
unzip ml-latest-small.zip -d data

# 2. install dependencies
pip install -r requirements.txt

# 3. open the notebook
jupyter notebook rekomendasi-film-movielens.ipynb
```

The notebook expects the dataset at `data/ml-latest-small/ratings.csv` and `data/ml-latest-small/movies.csv`.

## Possible next steps

- Matrix factorization (SVD) for latent factors
- Ranking evaluation (Precision@K) in addition to RMSE
- A small Streamlit demo where you pick favorite movies and get recommendations

Dataset credit: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1-19:19.
