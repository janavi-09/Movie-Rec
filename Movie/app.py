from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load data
movies_df = pd.read_csv("../data/movies.csv")
ratings_df = pd.read_csv("../data/ratings.csv")

# Function for Popularity-based recommendation
def popularity_based_recommendation(genre, min_reviews_threshold, num_recommendations):
    # Filter movies by genre
    genre_movies = movies_df[movies_df['genres'].str.contains(genre)]
    # Merge genre movies with ratings
    genre_ratings = ratings_df.merge(genre_movies, on='movieId')
    # Filter movies based on minimum reviews threshold
    genre_ratings_filtered = genre_ratings.groupby('movieId').filter(lambda x: len(x) >= min_reviews_threshold)
    # Calculate average rating and count of ratings for each movie
    popularity = genre_ratings_filtered.groupby('title').agg({'rating': 'mean', 'userId': 'count'}).reset_index()
    popularity.columns = ['title', 'average_rating', 'num_reviews']
    # Sort movies by average rating in descending order
    sorted_popularity = popularity.sort_values(by=['average_rating', 'num_reviews'], ascending=[False, False])
    # Get top N recommendations
    top_recommendations = sorted_popularity.head(num_recommendations)
    recommended_movie_titles = [f"{movie_title}|||  {avrg:.2f}|||  {num_rev}" for movie_title, avrg, num_rev in top_recommendations[['title', 'average_rating', 'num_reviews']].values]
    return recommended_movie_titles

# Function for Content-based recommendation
def content_based_recommendation(movie_title, num_recommendations):
    # Get movie ID for the given movie title
    movie_id = movies_df.loc[movies_df['title'] == movie_title, 'movieId'].iloc[0]
    # Get genres for the given movie
    input_genres = movies_df.loc[movies_df['movieId'] == movie_id, 'genres'].iloc[0]
    # Calculate TF-IDF matrix for movie genres
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(movies_df['genres'])
    # Calculate cosine similarities between movies based on genres
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    # Get index of the movie in the DataFrame
    movie_index = movies_df.index[movies_df['movieId'] == movie_id].tolist()[0]
    # Get similar movies based on cosine similarities
    similar_movies_scores = list(enumerate(cosine_similarities[movie_index]))
    similar_movies_scores_sorted = sorted(similar_movies_scores, key=lambda x: x[1], reverse=True)
    similar_movies_scores_sorted = [x for x in similar_movies_scores_sorted if x[0] != movie_index]
    # Get top N recommendations
    top_n_recommendations = similar_movies_scores_sorted[:num_recommendations]
    recommended_movie_titles = [f"{movies_df.iloc[idx]['title']}|||  {score:.2f}" for idx, score in top_n_recommendations]
    return recommended_movie_titles

# Function for Collaborative filtering recommendation
def collaborative_filtering_recommendation(user_id, num_recommendations, k_similar_users_threshold):
    # Filter ratings by target user ID
    target_user_ratings = ratings_df[ratings_df['userId'] == user_id]
    # Calculate average rating and count of ratings for each movie
    movie_ratings = ratings_df.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
    movie_ratings = movie_ratings.rename(columns={'mean': 'average_rating', 'count': 'num_ratings'})
    # Filter unrated movies based on minimum number of ratings threshold
    unrated_movies = pd.merge(movie_ratings, target_user_ratings['movieId'], on='movieId', how='left')
    unrated_movies['num_ratings'] = unrated_movies['num_ratings'].fillna(0)  # Fill NaN with 0 for unrated movies
    unrated_movies = unrated_movies[unrated_movies['num_ratings'] < k_similar_users_threshold]
    # Get ratings of similar users
    users_who_rated_unrated_movies = ratings_df[ratings_df['movieId'].isin(unrated_movies['movieId'])]
    user_movie_matrix = users_who_rated_unrated_movies.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)
    similarities = cosine_similarity([user_movie_matrix.loc[user_id]], user_movie_matrix)[0]
    similar_users_indices = np.argsort(similarities)[-k_similar_users_threshold:]
    similar_users_ratings = user_movie_matrix.iloc[similar_users_indices]
    average_ratings = similar_users_ratings.mean()
     # Get top N recommendations
    recommendations = average_ratings.sort_values(ascending=False)[:num_recommendations]
    recommended_movie_titles = [movies_df.loc[movies_df['movieId'] == movie_id, 'title'].iloc[0] for movie_id in recommendations.index]
    return recommended_movie_titles


# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/popularity_recommendations', methods=['POST'])
def popularity_recommendations():
    genre = request.form['genre']
    min_reviews_threshold = int(request.form['min_reviews_threshold'])
    num_recommendations = int(request.form['num_recommendations'])
    recommendations = popularity_based_recommendation(genre, min_reviews_threshold, num_recommendations)
    return render_template('popular.html', recommendations=recommendations)

@app.route('/content_recommendations', methods=['POST'])
def content_recommendations():
    movie_title = request.form['movie_title']
    num_recommendations = int(request.form['num_recommendations'])
    recommendations = content_based_recommendation(movie_title, num_recommendations)
    return render_template('content.html', recommendations=recommendations)

@app.route('/collaborative_filtering_recommendations', methods=['POST'])
def collaborative_filtering_recommendations():
    user_id = int(request.form['user_id'])
    num_recommendations = int(request.form['num_recommendations'])
    k_similar_users_threshold = int(request.form['k_similar_users_threshold'])
    recommendations = collaborative_filtering_recommendation(user_id, num_recommendations, k_similar_users_threshold)
    return render_template('collab.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
