from pprint import pprint
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import lil_matrix

MAX_MOVIE_ID = np.inf

movies = pd.read_csv('../ml-latest-small/movies.csv')
movies = movies[movies['movieId'] <= MAX_MOVIE_ID][['movieId', 'title']]
movie_ids = movies['movieId']
movie_indexes = dict(zip(movie_ids, np.arange(0, len(movie_ids))))
# print(movie_indexes)

ratings = pd.read_csv('../ml-latest-small/ratings.csv')
ratings = ratings[ratings['movieId'] <= MAX_MOVIE_ID]

# rows: users (don't care about their ids or anything), columns: movieIds
rating_matrix = ratings\
    .pivot(index='userId', columns='movieId', values='rating')\
    .reindex(movie_ids, axis='columns')\
    .fillna(0.0)

# create movie profile
user_ratings = lil_matrix((len(movie_indexes), 1), dtype=float)
user_ratings[movie_indexes[2571]] = 5.0  # Matrix
user_ratings[movie_indexes[32]] = 4.0  # Twelve Monkeys
user_ratings[movie_indexes[260]] = 5.0  # Star Wars IV
user_ratings[movie_indexes[1097]] = 4.0  # E.T. the Extra-Terrestrial

user_ratings = user_ratings.reshape(1, -1)
print(user_ratings)
user_profile = cosine_similarity(rating_matrix, user_ratings, dense_output=False)
user_profile_normalised = user_profile / np.linalg.norm(user_profile)
user_profile_normalised = user_profile_normalised.reshape(1, -1)

# recommendation
recommendation_vector = cosine_similarity(rating_matrix.T, user_profile_normalised, dense_output=False)
recommendation = [(similarity[0], movies.iloc[index]['title'])
                  for index, similarity in enumerate(recommendation_vector, start=0)
                  if similarity > 0]
recommendation.sort(key=lambda entry: entry[0], reverse=True)
print(f'Movies count: {len(movie_ids)}')
pprint(recommendation[:50], width=140)

