from pprint import pprint
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

MAX_MOVIE_ID = np.inf

movies = pd.read_csv('../ml-latest-small/movies.csv')
movies = movies[movies['movieId'] <= MAX_MOVIE_ID][['movieId', 'title']].set_index(movies['movieId'])
movie_ids = movies['movieId']

ratings = pd.read_csv('../ml-latest-small/ratings.csv')
ratings = ratings[ratings['movieId'] <= MAX_MOVIE_ID]

# rows: users (don't care about their ids or anything), columns: movieIds
rating_matrix = ratings\
    .pivot(index='userId', columns='movieId', values='rating')\
    .reindex(movie_ids, axis='columns', copy=False)\
    .fillna(0.0)

# create movie profile
user_ratings = pd.DataFrame(index=movie_ids, columns=['rating'])
user_ratings = user_ratings.fillna(0.0)
user_ratings.loc[2571, 'rating'] = 5.0  # Matrix
user_ratings.loc[32, 'rating'] = 4.0  # Twelve Monkeys
user_ratings.loc[260, 'rating'] = 5.0  # Star Wars IV
user_ratings.loc[1097, 'rating'] = 4.0  # E.T. the Extra-Terrestrial

user_profile = cosine_similarity(rating_matrix, [user_ratings['rating']], dense_output=False)
user_profile_normalised = user_profile / np.linalg.norm(user_profile)
user_profile_normalised = user_profile_normalised.reshape(1, -1)

# recommendation
recommendation_vector = cosine_similarity(rating_matrix.T, user_profile_normalised, dense_output=False)
recommendation_vector = pd.DataFrame(index=movie_ids, columns=['similarity'], data=recommendation_vector)
recommendation = [(entry['similarity'], *movies.loc[index])
                  for index, entry in recommendation_vector.iterrows()
                  ]
recommendation.sort(key=lambda entry: entry[0], reverse=True)
print(f'Movies count: {len(recommendation_vector)}')
pprint(recommendation[:50], width=140)

