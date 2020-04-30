import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

m = 10

# get id of 'Toy Story (1995)'
movies = pd.read_csv('ml-latest-small/movies.csv')
toy_story_id = movies[movies['title'] == 'Toy Story (1995)']['movieId'].to_numpy()[0]
print(f'Toy Story id: {toy_story_id}')

# get users (user Ids) who rated Toy Story
ratings = pd.read_csv('ml-latest-small/ratings.csv')
users = ratings[ratings['movieId'] == toy_story_id]
# print(users)

# get ratings for Toy Story
Y = users['rating'].to_numpy()
print(Y)
user_ids = users['userId'].to_numpy()
print(user_ids)

# c = ratings[(ratings['movieId'] <= m) & (ratings['userId'].isin(user_ids))][['userId', 'movieId', 'rating']]
c = ratings[(ratings['movieId'] <= m) & (ratings['userId'].isin(user_ids))]
print(c)
X = np.zeros(shape=(len(Y), m))
for i, u_id in enumerate(user_ids, 0):
    for j in range(0, m):
        g = c[(c['userId'] == u_id) & (c['movieId'] == j+2)]['rating'].to_numpy()
        print(g)
        if len(g) > 0:
            X[i][j] = g[0]
print(X)
print(X.shape)
clf = LinearRegression().fit(X, Y)
print(clf)


