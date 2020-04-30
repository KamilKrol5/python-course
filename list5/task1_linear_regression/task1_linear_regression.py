import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# get id of 'Toy Story (1995)'
movies = pd.read_csv('ml-latest-small/movies.csv')
toy_story_id = movies[movies['title'] == 'Toy Story (1995)']['movieId'].to_numpy()[0]
print(f'Toy Story id: {toy_story_id}')

# get users (user Ids) who rated Toy Story
ratings = pd.read_csv('ml-latest-small/ratings.csv')
users = ratings[ratings['movieId'] == toy_story_id][['rating', 'userId']].to_numpy()
Y = users[:, 0]
# print(Y)
# get all user ids
user_ids = users[:, 1]


def prepare_x(max_book_id):
    ratings_ = ratings[(ratings['movieId'] <= max_book_id) & (ratings['userId'].isin(user_ids))]
    X = ratings_.pivot(index='userId', columns='movieId', values='rating').to_numpy()[:, 1:]
    np.nan_to_num(X, copy=False)
    return X


chart_x = [10, 100, 500, 1000, 2500, 5000, 7500, 10000]
chart_y = []
for m in chart_x:
    X = prepare_x(m)
    clf = LinearRegression().fit(X, Y)
    chart_y.append(clf.score(X, Y))

plt.plot(chart_x, chart_y)
plt.xscale('log')
plt.show()
