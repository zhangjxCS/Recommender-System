import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Data preprocessing
# Import data
user = pd.read_csv("users.txt", names = ['userid'])
user['id'] = range(len(user))
netflix_train = pd.read_csv("netflix_train.txt", sep = ' ', names = ['user_id', 'film_id', 'rating', 'date'])
netflix_train = netflix_train.merge(user, left_on='user_id', right_on='userid')
netflix_test = pd.read_csv("netflix_test.txt", sep = ' ', names = ['user_id', 'film_id', 'rating', 'date'])
netflix_test = netflix_test.merge(user, left_on='user_id', right_on='userid')

X_train = netflix_train.pivot(index='id', columns='film_id', values='rating')
X_test = netflix_test.pivot(index='id', columns='film_id', values='rating')
print(X_train.head())
print(X_test.head())


# Collaborate Filtering
# Compute the overall mean and mean by row and column
mu = np.mean(np.mean(X_train))
bx = np.array(np.mean(X_train, axis=1) - mu)
by = np.array(np.mean(X_train, axis=0) - mu)
# Compute the similarity matrix
X = X_train.sub(bx+mu, axis=0)   # Demean
X = X.div(np.sqrt(np.sum(np.square(X), axis=1)), axis=0)
X.fillna(0, inplace=True)
similarity_matrix = np.dot(X, X.T)
# Compute the point matrix using CF
X_train = np.array(X_train.fillna(0))
for i in range(X_train.shape[0]):
    indexs = np.argsort(similarity_matrix[i, :])[::-1]
    for j in range(X_train.shape[1]):
        if X_train[i, j] == 0:
            sum = 0
            num = 0
            simi = 0
            k = 0
            while num < 3 & k < X_train.shape[1]:    # top 3
                if X_train[indexs[k], j] > 0:
                    sum = sum + similarity_matrix[i, indexs[k]] * (X_train[indexs[k], j] - mu - bx[indexs[k]] - by[j])
                    simi = simi + similarity_matrix[i, indexs[k]]
                    k = k+1
                    num = num + 1
                else:
                    k = k+1
            if simi != 0:
                X_train[i, j] = mu + bx[i] + by[j] + sum/simi
            else:
                X_train[i, j] = mu + bx[i] + by[j]
        else:
            continue
sum = 0
for index, rows in netflix_test.iterrows():
    sum = sum + np.square(X_train[rows['id'], rows['film_id']-1] - rows['rating'])
    print(X_train[rows['id'], rows['film_id']-1])
    print(sum)
RMSE = np.sqrt(sum/netflix_test.shape[0])
print(RMSE)
