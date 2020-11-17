# Recommender-System


数据集采用Netflix推荐竞赛的一个子集，包含10000个用户和10000个电影，具体的文件格式如下
(1) 用户列表 users.txt 
文件有 10000 行，每行一个整数，表示用户的 id，文件对应本次 Project 的所有用户。 
(2) 训练集 netflix_train.txt
文件包含 689 万条用户打分，每行为一次打分，对应的格式为: 用户 id 电影 id 分数 打分日期 其中用户 id 均出现在 users.txt 中，电影 id 为 1 到 10000 的整数。各项之间用空格分开 
(3) 测试集 netflix_test.txt
文件包含约 172 万条用户打分，格式与训练集相同。 
3.1 数据预处理
将输入文件整理成维度为用户*电影的矩阵 𝑋，其中𝑋𝑖𝑗对应用户 𝑖 对电影 𝑗 的打分

# 导入包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 导入数据
user = pd.read_csv("users.txt", names = ['userid'])
netflix_train = pd.read_csv("netflix_train.txt", sep = ' ', names = ['user_id', 'film_id', 'rating', 'date'])
netflix_test = pd.read_csv("netflix_test.txt", sep = ' ', names = ['user_id', 'film_id', 'rating', 'date'])
# 给用户从零开始进行编号
user['id'] = range(len(user))
netflix_train = netflix_train.merge(user, left_on='user_id', right_on='userid')
netflix_test = netflix_test.merge(user, left_on='user_id', right_on='userid')
# 通过数据透视函数构建用户*电影矩阵
X_train = netflix_train.pivot(index='id', columns='film_id', values='rating')
X_test = netflix_test.pivot(index='id', columns='film_id', values='rating')
# 测试集缺失部分电影，补齐为10000*10000矩阵
for i in range(1, 10001):
    if i not in X_test.columns:
        X_test[i] = np.nan
X_test = X_test.sort_index(axis=1)
# 查看输出的用户*电影矩阵
print(X_train.head())
print(X_test.head())

3.2 基于用户-用户协同过滤算法的实现
cosine相似度公式： cos(x,y) = \sqrt{\frac{x^T y}{|x||y|}}cos(x,y) = \sqrt{\frac{x^T y}{|x||y|}} 
评分计算： score(i, j)=\frac{\sum_k sim(X(i), X(k))score(k, j)}{\sum_k sim(X(i), X(k))}score(i, j)=\frac{\sum_k sim(X(i), X(k))score(k, j)}{\sum_k sim(X(i), X(k))} 
注意，此处对于未知值的计算，选择与该用户最相近的k个对此项目已评分的用户进行加权平均
（1）首先用argsort()函数求出与用户i最相似的用户，按照相似度倒序排列成列表indexs
（2）其次按照列表indexs进行遍历，找出看过此电影的相似度排名前三的用户并计算对电影评分的加权平均值作为该用户的评分
考虑到一些局部效应的存在，这里对原始算法进行了一些改进

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
# Compute RMSE for the algorithm
RMSE = np.sqrt(np.sum(np.sum(np.square(X_train - X_test)))/netflix_test.shape[0])
print(RMSE)

最终计算得到的RMSE为1.013，基线误差（预测得分全部取3的情况）为\sqrt2\sqrt2 ，RMSE降低了28.3%
3.3 基于矩阵分解的算法
矩阵分解： X_{m*n}\approx U_{m*k}V_{n*k}^TX_{m*n}\approx U_{m*k}V_{n*k}^T 
目标函数： J =\frac{1}{2}||A⊙(X-UV^T)||^2_F+\lambda||U||^2_F+\lambda||V||^2_FJ =\frac{1}{2}||A⊙(X-UV^T)||^2_F+\lambda||U||^2_F+\lambda||V||^2_F 
\frac{\partial J}{\partial U}=(A⊙(UV^T-X))V+2\lambda U\frac{\partial J}{\partial U}=(A⊙(UV^T-X))V+2\lambda U 
\frac{\partial J}{\partial V} = (A⊙(UV^T-X))^TU+2\lambda V\frac{\partial J}{\partial V} = (A⊙(UV^T-X))^TU+2\lambda V 
通过梯度下降算法迭代更新目标函数，获取最优分解矩阵U和V

# Matrix Decomposition
A = X_train > 0
X_train = np.array(X_train.fillna(0))
U = np.random.randn(10000, 100)*0.1
V = np.random.randn(10000, 100)*0.1
alpha = 0.0001
lamda = 1
# Gradient Descent
J = np.zeros((1000))
RMSE = np.zeros((1000))
for i in range(200):
    dU = np.dot(np.multiply(A, (np.dot(U, V.T) - X_train)), V) + 2 * lamda * U
    dV = np.dot(np.multiply(A, (np.dot(U, V.T) - X_train)), U) + 2 * lamda * V
    old_U = U
    old_V = V
    U = U - alpha/(1+0.1*i) * dU # Learning rate decay
    V = V - alpha/(1+0.1*i) * dV
    J[i, 0] = 1/2*np.sum(np.sum(np.square(np.multiply(A, (X_train - np.dot(U, V.T)))))) + lamda * np.sum(np.sum(np.square(U)))\
           + lamda * np.sum(np.sum(np.square(V)))
    RMSE[i, 0] = np.sqrt(np.sum(np.sum(np.square(np.dot(U, V.T) - X_test)))/netflix_test.shape[0])
    print(i)
# Visualization
X = np.dot(U, V.T)
plt.plot(range(1000), RMSE[:, 0])
plt.show()
plt.plot(range(1000), J[:, 0])
plt.show()
print(RMSE[999])

矩阵分解的算法收敛效果与模型中的正则项系数 \lambda\lambda 与矩阵维度k是有关，可以尝试不同的参数组合，通过RMSE与目标函数值来确定最优参数组合，目标函数值随着迭代次数的变化如下图所示，尝试的参数组合分别为 \lambda\lambda =1, 0.1 以及 矩阵维数k=100, 50, 10 共2*3=6种，每种参数组合迭代200次，迭代结果如下图所示，可以选择收敛最快的参数组合进行训练
4.总结
从上文我们可以看出推荐系统的核心问题是确定用户-内容矩阵(Utility Matrix)
(1)收集已知矩阵信息
通过让用户打分或者收集用户的行为数据
(2)从已知矩阵推测未知矩阵信息
通过基于内容的方法、协同过滤方法或者矩阵分解方法推测未知矩阵信息
(3)评价推测方法
常用的标准是RMSE均方根误差
