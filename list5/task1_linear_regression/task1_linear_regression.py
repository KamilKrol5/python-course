import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# get id of 'Toy Story (1995)'
movies = pd.read_csv('ml-latest-small/movies.csv')
toy_story_id = movies[movies['title'] == 'Toy Story (1995)']['movieId'].to_numpy()[0]
print(f'Toy Story id: {toy_story_id}')

# get users (user Ids) who rated Toy Story
ratings = pd.read_csv('ml-latest-small/ratings.csv')
users = ratings[ratings['movieId'] == toy_story_id][['rating', 'userId']].to_numpy()
Y = users[:, 0]
print(Y)
# get all user ids
user_ids = users[:, 1]


def prepare_x(m):
    ratings_ = ratings[(ratings['movieId'] <= m) & (ratings['userId'].isin(user_ids))]
    X = ratings_.pivot(index='userId', columns='movieId', values='rating').to_numpy()[:, 1:]
    np.nan_to_num(X, copy=False)
    return X


for m in [10,100,1000,10000]:
    X = prepare_x(m)
    clf = LinearRegression().fit(X, Y)
    print(clf.score(X, Y))


