from pprint import pprint
from typing import Union, List

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import lil_matrix
import dask.dataframe as ddf


def load_data(filename: str, columns: Union[List[str], List[int]]) -> ddf.DataFrame:
    return ddf.read_csv(filename, usecols=columns, blocksize=1024 * 1024)


def compute_users_similarity(data_for_single_user: pd.DataFrame) -> object:
    data_for_single_user = data_for_single_user.set_index('movieId')
    data_for_single_user = data_for_single_user.reindex(movies.index).fillna(0.0)
    return cosine_similarity(data_for_single_user.T, user_ratings.T)[0, 0]


def group_movies_recommendation(data_for_single_movie: pd.DataFrame) -> object:
    data_for_single_movie = data_for_single_movie.set_index('userId')
    data_for_single_movie = data_for_single_movie.reindex(profile.index).fillna(0.0)
    return cosine_similarity(data_for_single_movie.T, profile.T)[0, 0]


movies = load_data('../ml-latest-small/movies.csv', columns=['movieId', 'title'])
movies = movies.set_index('movieId', sorted=True)
movies: ddf.DataFrame = movies.loc[:10000].compute()

ratings = load_data('../ml-latest-small/ratings.csv', columns=['userId', 'movieId', 'rating'])
ratings: ddf.DataFrame = ratings.set_index('userId', sorted=True)
ratings = ratings.repartition(ratings.divisions)

user_ratings = {
    2571: 5.0,  # Matrix
    32: 4.0,  # Twelve Monkeys
    260: 5.0,  # Star Wars IV
    1097: 4.0  # E.T. the Extra-Terrestrial
}
user_ratings = pd.DataFrame.from_dict(user_ratings, orient='index', columns=['rating'])
user_ratings = user_ratings.reindex(movies.index).fillna(0.0)
profile = ratings.groupby('userId').apply(compute_users_similarity, meta=object)
profile = pd.DataFrame(profile)

ratings.reset_index().set_index('movieId').to_hdf('./ratings_by_movie_id.hdf', 'movieId')
recommendation_vector = ddf.read_hdf('./ratings_by_movie_id.hdf', 'movieId', columns=['userId', 'rating'], sorted_index=True)
recommendation_vector2 = recommendation_vector.groupby('movieId').apply(group_movies_recommendation, meta=object)