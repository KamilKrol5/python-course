import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# get id of 'Toy Story (1995)'
movies = pd.read_csv('../ml-latest-small/movies.csv')
toy_story_id = movies[movies['title'] == 'Toy Story (1995)']['movieId'].to_numpy()[0]
movie_ids = movies['movieId'].to_numpy()
print(f'Toy Story id: {toy_story_id}')

# get users (user Ids) who rated Toy Story
ratings = pd.read_csv('../ml-latest-small/ratings.csv')
users = ratings[ratings['movieId'] == toy_story_id][['rating', 'userId']].to_numpy()
Y = users[:, 0]
# print(Y)
# get all user ids
user_ids = users[:, 1]


def prepare_x(max_book_id, usr_ids):
    indexes = movie_ids[movie_ids <= max_book_id]
    ratings_ = ratings[(ratings['movieId'] <= max_book_id) & (ratings['userId'].isin(usr_ids))]
    X_ = ratings_.pivot(index='userId', columns='movieId', values='rating').reindex(indexes[1:], axis='columns')

    np.nan_to_num(X_, copy=False)
    return X_


chart_x = [10, 100, 200, 500, 1000, 2500, 5000, 7500, 10000]
chart_y = []
for m in chart_x:
    X = prepare_x(m, user_ids)
    # print(X)
    clf = LinearRegression().fit(X, Y)
    chart_y.append(clf.score(X, Y))

plt.plot(chart_x, chart_y, linewidth=8)
plt.xscale('log')
plt.xlabel('m')
plt.ylabel('determination R^2')
plt.show()

# learning
for m in chart_x:
    X = prepare_x(m, user_ids[:-15])
    clf = LinearRegression().fit(X, Y[:-15])
    # prediction
    X_for_prediction = prepare_x(m, user_ids[-15:])
    predicted = clf.predict(X_for_prediction)
    # print(f'Predicted: {predicted}')
    # print(f'Expected:  {Y[-15:]}')
    print(f'---m = {m}---\n(Expected, Predicted):\n{" ".join([str(pair) for pair in zip(Y[-15:], predicted)])}')

    if m == 10000:
        fig, ax = plt.subplots(1, 1)
        ax.grid(True)
        ax.scatter(np.arange(1, 16), predicted, c='coral', s=30, label='predicted')
        ax.scatter(np.arange(1, 16), Y[-15:], c='green', s=50, label='expected')
        ax.set_title('Regression model prediction results (m=10000)')
        ax.legend()
        fig.tight_layout()
        plt.show()
