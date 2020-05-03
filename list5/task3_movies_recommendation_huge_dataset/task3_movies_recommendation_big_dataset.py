from pprint import pprint
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import lil_matrix
import dask.dataframe as ddf

# prepare data
movies = pd.read_csv('../ml-latest-small/movies.csv')
movies = movies[['movieId', 'title']]
movie_ids = movies['movieId'].to_numpy()
types = {'userId': int,
         'movieId': int,
         'rating': pd.SparseDtype(dtype='float32', fill_value=0.0)}

chunks = ddf.read_csv('../ml-latest-small/ratings.csv')
chunks = chunks.map_partitions(
    lambda part:
    part[['userId', 'movieId', 'rating']].astype(types, copy=False))
ratings = chunks.compute()
rating_matrix = ratings \
    .pivot_table(index='userId', columns='movieId', values='rating') \
    .fillna(0.0)

# create movie profile
user_ratings = lil_matrix((len(np.unique(ratings['movieId'])), 1))
movie_indexes = dict(zip(np.unique(ratings['movieId']), np.arange(0, len(np.unique(ratings['movieId'])))))
user_ratings[movie_indexes[2571]] = 5.0  # Matrix
user_ratings[movie_indexes[32]] = 4.0  # Twelve Monkeys
user_ratings[movie_indexes[260]] = 5.0  # Star Wars IV
user_ratings[movie_indexes[1097]] = 4.0  # E.T. the Extra-Terrestrial
user_ratings = user_ratings.reshape(1, -1)

user_profile = cosine_similarity(rating_matrix, user_ratings, dense_output=False)
user_profile_normalised = user_profile / np.linalg.norm(user_profile)
user_profile_normalised = user_profile_normalised.reshape(1, -1)

# recommendation
recommendation_vector = cosine_similarity(rating_matrix.T, user_profile_normalised, dense_output=False)
recommendation = [(similarity[0],  movies.iloc[index]['movieId'], movies.iloc[index]['title'])
                  for index, similarity in enumerate(recommendation_vector)
                  if similarity[0] > 0]
recommendation.sort(key=lambda entry: entry[0], reverse=True)
print(f'Movies count: {len(np.unique(ratings["movieId"]))}')
pprint(recommendation[:50], width=140)
