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

# Matrix Decomposition
A = X_train > 0
X_train = np.array(X_train.fillna(0))
U1 = np.random.randn(10000, 100)*0.1
V1 = np.random.randn(10000, 100)*0.1
U2 = np.random.randn(10000, 50)*0.1
V2 = np.random.randn(10000, 50)*0.1
U3 = np.random.randn(10000, 10)*0.1
V3 = np.random.randn(10000, 10)*0.1
U4 = np.random.randn(10000, 100)*0.1
V4 = np.random.randn(10000, 100)*0.1
U5 = np.random.randn(10000, 50)*0.1
V5 = np.random.randn(10000, 50)*0.1
U6 = np.random.randn(10000, 10)*0.1
V6 = np.random.randn(10000, 10)*0.1
alpha = 0.0001
lamda1 = 1
lamda2 = 0.1

J = np.zeros((200, 6))
# U1 V1 lamda1
for i in range(200):
    dU = np.dot(np.multiply(A, (np.dot(U1, V1.T) - X_train)), V1) + 2 * lamda1 * U1
    dV = np.dot(np.multiply(A, (np.dot(U1, V1.T) - X_train)), U1) + 2 * lamda1 * V1
    old_U = U1
    old_V = V1
    U1 = U1 - alpha/(1+0.1*i) * dU # Learning rate decay
    V1 = V1 - alpha/(1+0.1*i) * dV
    J[i, 0] = 1/2*np.sum(np.sum(np.square(np.multiply(A, (X_train - np.dot(U1, V1.T)))))) + lamda1 * np.sum(np.sum(np.square(U1)))\
           + lamda1 * np.sum(np.sum(np.square(V1)))
    print(i)

# U2 V2 lamda1
for i in range(200):
    dU = np.dot(np.multiply(A, (np.dot(U2, V2.T) - X_train)), V2) + 2 * lamda1 * U2
    dV = np.dot(np.multiply(A, (np.dot(U2, V2.T) - X_train)), U2) + 2 * lamda1 * V2
    old_U = U2
    old_V = V2
    U2 = U2 - alpha/(1+0.1*i) * dU # Learning rate decay
    V2 = V2 - alpha/(1+0.1*i) * dV
    J[i, 1] = 1/2*np.sum(np.sum(np.square(np.multiply(A, (X_train - np.dot(U2, V2.T)))))) + lamda1 * np.sum(np.sum(np.square(U2)))\
           + lamda1 * np.sum(np.sum(np.square(V2)))
    print(i)

# U3 V3 lamda1
for i in range(200):
    dU = np.dot(np.multiply(A, (np.dot(U3, V3.T) - X_train)), V3) + 2 * lamda1 * U3
    dV = np.dot(np.multiply(A, (np.dot(U3, V3.T) - X_train)), U3) + 2 * lamda1 * V3
    old_U = U3
    old_V = V3
    U3 = U3 - alpha/(1+0.1*i) * dU # Learning rate decay
    V3 = V3 - alpha/(1+0.1*i) * dV
    J[i, 2] = 1/2*np.sum(np.sum(np.square(np.multiply(A, (X_train - np.dot(U3, V3.T)))))) + lamda1 * np.sum(np.sum(np.square(U3)))\
           + lamda1 * np.sum(np.sum(np.square(V3)))
    print(i)

# U4 V4 lamda2
for i in range(200):
    dU = np.dot(np.multiply(A, (np.dot(U4, V4.T) - X_train)), V4) + 2 * lamda2 * U4
    dV = np.dot(np.multiply(A, (np.dot(U4, V4.T) - X_train)), U4) + 2 * lamda2 * V4
    old_U = U4
    old_V = V4
    U4 = U4 - alpha/(1+0.1*i) * dU # Learning rate decay
    V4 = V4 - alpha/(1+0.1*i) * dV
    J[i, 3] = 1/2*np.sum(np.sum(np.square(np.multiply(A, (X_train - np.dot(U4, V4.T)))))) + lamda2 * np.sum(np.sum(np.square(U4)))\
           + lamda2 * np.sum(np.sum(np.square(V4)))
    print(i)

# U5 V5 lamda2
for i in range(200):
    dU = np.dot(np.multiply(A, (np.dot(U5, V5.T) - X_train)), V5) + 2 * lamda2 * U5
    dV = np.dot(np.multiply(A, (np.dot(U5, V5.T) - X_train)), U5) + 2 * lamda2 * V5
    old_U = U5
    old_V = V5
    U5 = U5 - alpha/(1+0.1*i) * dU # Learning rate decay
    V5 = V5 - alpha/(1+0.1*i) * dV
    J[i, 4] = 1/2*np.sum(np.sum(np.square(np.multiply(A, (X_train - np.dot(U5, V5.T)))))) + lamda2 * np.sum(np.sum(np.square(U5)))\
           + lamda2 * np.sum(np.sum(np.square(V5)))
    print(i)

# U6 V6 lamda2
for i in range(200):
    dU = np.dot(np.multiply(A, (np.dot(U6, V6.T) - X_train)), V6) + 2 * lamda2 * U6
    dV = np.dot(np.multiply(A, (np.dot(U6, V6.T) - X_train)), U6) + 2 * lamda2 * V6
    old_U = U6
    old_V = V6
    U6 = U6 - alpha/(1+0.1*i) * dU # Learning rate decay
    V6 = V6 - alpha/(1+0.1*i) * dV
    J[i, 5] = 1/2*np.sum(np.sum(np.square(np.multiply(A, (X_train - np.dot(U6, V6.T)))))) + lamda2 * np.sum(np.sum(np.square(U6)))\
           + lamda2 * np.sum(np.sum(np.square(V6)))
    print(i)

plt.plot(range(200), J[:, 0], label='100, 1')
plt.plot(range(200), J[:, 1], label='50, 1')
plt.plot(range(200), J[:, 2], label='10, 1')
plt.plot(range(200), J[:, 3], label='100, 0.1')
plt.plot(range(200), J[:, 4], label='50, 0.1')
plt.plot(range(200), J[:, 5], label='10, 0.1')
plt.legend()
plt.show()

X = np.dot(U4, V4.T)
sum = 0
for index, rows in netflix_test.iterrows():
    sum = sum + np.square(X[rows['id'], rows['film_id']-1] - rows['rating'])
    print(X[rows['id'], rows['film_id']-1])
    print(sum)
RMSE = np.sqrt(sum/netflix_test.shape[0])
print(RMSE)
