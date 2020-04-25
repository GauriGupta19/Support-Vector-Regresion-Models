#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.svm import SVC
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from sklearn import svm
from sklearn.model_selection import train_test_split as tts
import random
from sklearn import metrics 


# In[3]:


#normalising the data set
df = pd.read_csv('data.csv')
normalized_df=(df-df.min())/(df.max()-df.min())
data = np.array(normalized_df)
X = data[:, 1:-1]
y = data[:, -1]
m,n = X.shape
print(type(y[0]))
print(y.shape)
print(X.shape)


# In[4]:


#kernel functions used
def kernel(kernel_type, A1, A2, gamma, degree):
    if kernel_type == 'linear':
        return np.dot(A1, A2.T)
    if kernel_type == 'rbf':
        K = np.zeros((A1.shape[0], A2.shape[0]))
        for i, a1 in enumerate(A1):
            for j, a2 in enumerate(A2):
                K[i, j] = np.exp(-gamma * np.linalg.norm(a1 - a2) ** 2)
        return K
    if kernel_type == 'poly':
        K = np.zeros((A1.shape[0], A2.shape[0]))
        for i, a1 in enumerate(A1):
            for j, a2 in enumerate(A2):
                K[i, j] = gamma*((a1@a2.T + 1) ** degree)
        return K
        
    


# # Epsilon SVR

# In[5]:


#formulating eps-SVR
def svr_cvxopt(X_train, y_train, X_test, C, eps, kernel_type, gamma, degree):
    X = X_train
    y = y_train
    m,n = X.shape


    y = y.reshape(-1,1) * 1
    print(y.shape)
    q1 = eps - y
    q2 = eps + y
    K = kernel(kernel_type, X, X, gamma, degree)
    p1 = np.hstack((K, K*-1))
    P = cvxopt_matrix(np.vstack((p1, p1*-1)))
    q = cvxopt_matrix(np.vstack((q1, q2)))
    G = cvxopt_matrix(np.vstack((np.eye(2*m)*-1, np.eye(2*m))))
    h = cvxopt_matrix(np.hstack((np.zeros(2*m), np.ones(2*m) * C)))
    A = cvxopt_matrix((np.hstack((np.ones(m), np.ones(m)*-1))).reshape(1,-1))
    # print(A.shape)
    b = cvxopt_matrix(np.zeros(1))

    #Run solver
    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    # print(alphas)
    alphas1 = alphas[:m]
    alphas2 = alphas[m:2*m]
    b = np.array(sol['y'])
#     print(b)
#     W = ((alphas1 - alphas2).T @ X).reshape(-1, 1) ## W is a column vector
    y_pred = kernel(kernel_type, X, X_test, gamma, degree).T@(alphas1 - alphas2) + b
#     print(y_pred)
    return y_pred


# In[6]:


# k-fold cross validation
# returns MSE for both cvxopt and sklearn on gthe given dataset
def cross_val(X, y, num_folds, C, eps, kernel_type, gamma, degree):
    y = y.reshape(-1,1) * 1
    m, n = X.shape
    batch_size = m//num_folds

    mse1 = 0
    mse2 = 0
    for i in range(1, num_folds+1):
        X_test = X[(i-1)*batch_size: i*batch_size, :]
        y_test = y[(i-1)*batch_size: i*batch_size:, ]
        if i == 1:
            X_train = X[i*batch_size : , :]
            y_train = y[i*batch_size :, :]
        else:
            X_train = X[: (i-1)*batch_size, :]
            np.vstack((X_train, X[i*batch_size : , :]))
            y_train = y[: (i-1)*batch_size, :]
            np.vstack((y_train, y[i*batch_size : , :]))
        reg = svm.SVR(kernel = kernel_type, gamma = gamma, degree = degree,  C = C, epsilon = eps).fit(X_train, y_train.ravel()) 
        y_sk = reg.predict(X_test)
        
        y_pred = svr_cvxopt(X_train, y_train, X_test, C, eps, kernel_type, gamma, degree)
        y_pred = y_pred[:, 0]
        y_test = y_test[:, 0]
#         print(y_pred.shape)
#         print(y_test.shape)
#         print(y_sk.shape)
#         print(np.max(y_pred-y_test))
#         print(np.min(y_pred-y_test))
        mse1 += (np.linalg.norm(y_pred - y_test) ** 2)/y_test.shape[0]
        mse2 += (np.linalg.norm(y_sk - y_test) ** 2)/y_test.shape[0]

#         score1 += metrics.r2_score(y_test,y_pred)
#         score2 += metrics.r2_score(y_test,y_sk)

    mse1 = mse1/num_folds
    mse2 = mse2/num_folds
    return mse1, mse2
    


# In[7]:


#k-fold cross validation , retuns the MSE for cvxopt implemantation
def cross_val_cvx(X, y, num_folds, C, eps, kernel_type, gamma, degree):
    y = y.reshape(-1,1) * 1
    m, n = X.shape
    batch_size = m//num_folds

    mse1 = 0
    for i in range(1, num_folds+1):
        X_test = X[(i-1)*batch_size: i*batch_size, :]
        y_test = y[(i-1)*batch_size: i*batch_size:, ]
        if i == 1:
            X_train = X[i*batch_size : , :]
            y_train = y[i*batch_size :, :]
        else:
            X_train = X[: (i-1)*batch_size, :]
            np.vstack((X_train, X[i*batch_size : , :]))
            y_train = y[: (i-1)*batch_size, :]
            np.vstack((y_train, y[i*batch_size : , :]))
        
        y_pred = svr_cvxopt(X_train, y_train, X_test, C, eps, kernel_type, gamma, degree)
        y_pred = y_pred[:, 0]
        y_test = y_test[:, 0]

        mse1 += (np.linalg.norm(y_pred - y_test) ** 2)/y_test.shape[0]
        
    mse1 = mse1/num_folds

    return mse1
    


# In[8]:


#k-fold cross validation , retuns the MSE for sklearn implemantation
def cross_val_sk(X, y, num_folds, C, eps, kernel_type, gamma, degree):
    y = y.reshape(-1,1) * 1
    m, n = X.shape
    batch_size = m//num_folds

    mse = 0

    for i in range(1, num_folds+1):
        X_test = X[(i-1)*batch_size: i*batch_size, :]
        y_test = y[(i-1)*batch_size: i*batch_size:, ]
        if i == 1:
            X_train = X[i*batch_size : , :]
            y_train = y[i*batch_size :, :]
        else:
            X_train = X[: (i-1)*batch_size, :]
            np.vstack((X_train, X[i*batch_size : , :]))
            y_train = y[: (i-1)*batch_size, :]
            np.vstack((y_train, y[i*batch_size : , :]))
        reg = svm.SVR(kernel = kernel_type, gamma = gamma, degree = degree,  C = C, epsilon = eps).fit(X_train, y_train.ravel()) 
        y_sk = reg.predict(X_test)
        y_test = y_test[:, 0]
        mse += (np.linalg.norm(y_sk - y_test) ** 2)/y_test.shape[0]

    mse = mse/num_folds
    return mse
    


# # Linear Kernel

# In[6]:


#plot of the predicted MEDV values vs the true values for the linear kernel
X_train = X[101: , :]
y_train = y[101: ]
X_test = X[:101, :]
y_test = y[:101]
reg = svm.SVR(kernel = 'linear', degree = 2, C = 1, epsilon = 0.1).fit(X_train, y_train.ravel()) 
y_sk = reg.predict(X_test)
# score1 = reg.score(X_test, y_test)
y_pred = svr_cvxopt(X_train, y_train, X_test, 1, 0.1, 'linear', 1, 2)
# print(y_pred)
# print(y_sk)
# print(y)
y_pred = y_pred[:, 0]
print(y_pred.shape)
print((np.linalg.norm(y_pred - y_test) ** 2)/y_test.shape[0])
print((np.linalg.norm(y_sk - y_test) ** 2)/y_test.shape[0])
# print(np.max(y_sk-y_test))

Xx = np.array(list(range(0, y_test.shape[0]))).reshape(y_test.shape[0], 1)
plt.plot(Xx, y_test, label ='True')
plt.plot(Xx, y_pred, label ='Predicted cvxopt')

plt.plot(Xx, y_sk, label ='Predicted Sklearn')
plt.xlabel('Data Points')
plt.ylabel('MEDV')
plt.legend()
plt.title('linear Kernel')
plt.show()


# In[16]:


#plot of the MSE vs the hyperparamter C for linear kernel
C_values = np.linspace(1, 100, 10, endpoint=True)
mse_cvxopt = []
mse_sk = []
# mse_sklearn = []
for i in C_values:         
    mse1, mse2 = cross_val(X, y, 5, i, 0.1, 'linear', 1, 1)
    mse_cvxopt.append(mse1)
    mse_sk.append(mse2)

print(mse_cvxopt)
print(mse_sk)   
plt.scatter(C_values, mse_cvxopt, label ='cvxopt')
plt.scatter(C_values, mse_sk, label ='sklearn')

plt.xlabel(' Value of C')
plt.ylabel('MSE')
plt.legend()
plt.title('Linear kernel')
plt.show()
# plt.savefig('MEDV vs data.png')


# In[17]:


#plot of the MSE vs the eps for linear kernel
x_values = np.linspace(0.01, 1, 10, endpoint=True)
mse_cvxopt = []
mse_sk = []
# mse_sklearn = []
for i in x_values:         
    mse1, mse2 = cross_val(X, y, 5, 1, i, 'linear', 1, 1)
    mse_cvxopt.append(mse1)
    mse_sk.append(mse2)

print(mse_cvxopt)
print(mse_sk)   
plt.scatter(x_values, mse_cvxopt, label ='cvxopt')
plt.scatter(x_values, mse_sk, label ='sklearn')

plt.xlabel(' Value of eps')
plt.ylabel('MSE')
plt.legend()
plt.title('Linear kernel')
plt.show()
# plt.savefig('MEDV vs data.png')


# # RBF Kernel

# In[12]:


#plot of the predicted MEDV values vs the true values for the rbf kernel
X_train = X[101: , :]
y_train = y[101: ]
X_test = X[:101, :]
y_test = y[:101]
reg = svm.SVR(kernel = 'rbf', gamma = 1, C = 1, epsilon = 0.1).fit(X_train, y_train.ravel()) 
y_sk = reg.predict(X_test)
# score1 = reg.score(X_test, y_test)
y_pred = svr_cvxopt(X_train, y_train, X_test, 1, 0.1, 'rbf', 1, 2)
# print(y_pred)
# print(y_sk)
# print(y)
y_pred = y_pred[:, 0]
print(y_pred.shape)
print((np.linalg.norm(y_pred - y_test) ** 2)/y_test.shape[0])
print((np.linalg.norm(y_sk - y_test) ** 2)/y_test.shape[0])
# print(np.max(y_sk-y_test))

Xx = np.array(list(range(0, y_test.shape[0]))).reshape(y_test.shape[0], 1)
plt.plot(Xx, y_test, label ='True')
plt.plot(Xx, y_pred, label ='Predicted cvxopt')

plt.plot(Xx, y_sk, label ='Predicted Sklearn')
plt.xlabel('Data Points')
plt.ylabel('MEDV')
plt.legend()
plt.title('Rbf Kernel')
plt.show()


# In[20]:


#plot of the MSE vs the hyperparamter C for rbf kernel
x_values = np.linspace(1, 100, 10, endpoint=True)
mse_cvxopt = []
mse_sk = []
# mse_sklearn = []
for i in x_values:         
    mse1, mse2 = cross_val(X, y, 5, i, 0.1, 'rbf', 1, 1)
    mse_cvxopt.append(mse1)
    mse_sk.append(mse2)

print(mse_cvxopt)
print(mse_sk)   
plt.scatter(x_values, mse_cvxopt, label ='cvxopt')
plt.scatter(x_values, mse_sk, label ='sklearn')

plt.xlabel(' Value of C')
plt.ylabel('MSE')
plt.legend()
plt.title('rbf kernel')
plt.show()
# plt.savefig('MEDV vs data.png')


# In[22]:


#plot of the MSE vs the hyperparamter gamma for rbf kernel
x_values = np.linspace(0.1, 10, 10, endpoint=True)
mse_cvxopt = []
mse_sk = []
# mse_sklearn = []
for i in x_values:         
    mse1, mse2 = cross_val(X, y, 5, 1, 0.1, 'rbf', i, 1)
    mse_cvxopt.append(mse1)
    mse_sk.append(mse2)

print(mse_cvxopt)
print(mse_sk)   
plt.scatter(x_values, mse_cvxopt, label ='cvxopt')
plt.scatter(x_values, mse_sk, label ='sklearn')

plt.xlabel(' Value of gamma')
plt.ylabel('MSE')
plt.legend()
plt.title('rbf kernel')
plt.show()
# plt.savefig('MEDV vs data.png')


# In[28]:


#finding the best hyperparameters for rbf kernel
min_error = 100
best_c = 0.1
best_gamma = 1
C_values = np.linspace(0.1, 10, 10)
gamma_values = np.linspace(0.01, 1, 10)
ac_mat = np.zeros([len(C_values), len(gamma_values)])
Xx, Yy = np.meshgrid(gamma_values, C_values)
for i in range(len(C_values)):
    for j in range(len(gamma_values)):        
        mse = cross_val_cvx(X, y, 5, C_values[i], 0.1, 'rbf', gamma_values[j], 1)
        if mse < min_error:
            min_error = mse
            best_c = C_values[i]
            best_gamma = gamma_values[j]
        ac_mat[i, j] = mse
print('cvxopt mse : ', min_error, 'best_c : ', best_c, 'best_gamma : ', best_gamma)
mse_sk = cross_val_sk(X, y, 5, best_c, 1, 'rbf', best_gamma, 1)
print('sklearn mse : ', mse_sk)
# print(np.sort(ans.support_))
# print(ans.support_.shape)

plt.figure(figsize=(9.5,6.5))
ax = plt.axes(projection='3d')
ax.plot_surface(Xx, Yy, ac_mat, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_xlabel('Value of Gamma')
ax.set_ylabel('Value of C')
ax.set_zlabel('MSE');
plt.show()
plt.contourf(Xx, Yy, ac_mat)
plt.colorbar()
plt.xlabel('Value of Gamma')
plt.ylabel('Value of C')
plt.show()


# # Polynomial kernel
# 

# In[14]:


#plot of the predicted MEDV values vs the true values for the polynomial kernel with degree = 2
X_train = X[101: , :]
y_train = y[101: ]
X_test = X[:101, :]
y_test = y[:101]
reg = svm.SVR(kernel = 'poly', gamma = 1, degree = 2, C = 1, epsilon = 0.1).fit(X_train, y_train.ravel()) 
y_sk = reg.predict(X_test)
# score1 = reg.score(X_test, y_test)
y_pred = svr_cvxopt(X_train, y_train, X_test, 1, 0.1, 'poly', 1, 2)
# print(y_pred)
# print(y_sk)
# print(y)
y_pred = y_pred[:, 0]
print(y_pred.shape)
print((np.linalg.norm(y_pred - y_test) ** 2)/y_test.shape[0])
print((np.linalg.norm(y_sk - y_test) ** 2)/y_test.shape[0])
# print(np.max(y_sk-y_test))

Xx = np.array(list(range(0, y_test.shape[0]))).reshape(y_test.shape[0], 1)
plt.plot(Xx, y_test, label ='True')
plt.plot(Xx, y_pred, label ='Predicted cvxopt')

plt.plot(Xx, y_sk, label ='Predicted Sklearn')
plt.xlabel('Data Points')
plt.ylabel('MEDV')
plt.legend()
plt.title('Polynomial Kernel')
plt.show()


# In[27]:


#plot of the predicted MEDV values vs the true values for the polynomial kernel with degree = 2
X_train = X[101: , :]
y_train = y[101: ]
X_test = X[:101, :]
y_test = y[:101]
reg = svm.SVR(kernel = 'poly', gamma = 1, degree = 5, C = 1, epsilon = 0.1).fit(X_train, y_train.ravel()) 
# y_sk = reg.predict(X_test)
# score1 = reg.score(X_test, y_test)
y_pred = svr_cvxopt(X_train, y_train, X_test, 10, 0.01, 'poly', 1, 6)
# print(y_pred)
# print(y_sk)
# print(y)
y_pred = y_pred[:, 0]
print(y_pred.shape)
# print((np.linalg.norm(y_pred - y_test) ** 2)/y_test.shape[0])
# print((np.linalg.norm(y_sk - y_test) ** 2)/y_test.shape[0])
# # print(np.max(y_sk-y_test))

Xx = np.array(list(range(0, y_test.shape[0]))).reshape(y_test.shape[0], 1)
plt.plot(Xx, y_test, label ='True')
plt.plot(Xx, y_pred, label ='Predicted cvxopt')

# plt.plot(Xx, y_sk, label ='Predicted Sklearn')
plt.xlabel('Data Points')
plt.ylabel('MEDV')
plt.legend()
plt.title('Polynomial Kernel')
plt.show()


# In[29]:


#plot of the predicted MEDV values vs the true values for the polynomial kernel with degree = 2
X_train = X
y_train = y
X_test = X[:101, :]
y_test = y[:101]
# X_train = X[:101, :]
# y_train = y[:101]
# reg = svm.SVR(kernel = 'poly', gamma = 1, degree = 5, C = 1, epsilon = 0.1).fit(X_train, y_train.ravel()) 
# y_sk = reg.predict(X_train)
# score1 = reg.score(X_test, y_test)
y_pred = svr_cvxopt(X_train, y_train, X_test, 1, 0.1, 'poly', 1, 6)
# print(y_pred)
# print(y_sk)
# print(y)
y_pred = y_pred[:, 0]
# print(y_pred.shape)
# print((np.linalg.norm(y_pred - y_train) ** 2)/y_train.shape[0])
# print((np.linalg.norm(y_sk - y_train) ** 2)/y_train.shape[0])
# print(np.max(y_sk-y_test))

Xx = np.array(list(range(0, y_test.shape[0]))).reshape(y_test.shape[0], 1)
plt.plot(Xx, y_test, label ='True')
plt.plot(Xx, y_test, label ='Predicted cvxopt')

# plt.plot(Xx, y_sk, label ='Predicted Sklearn')
plt.xlabel('train data')
plt.ylabel('MEDV')
plt.legend()
plt.title('Polynomial Kernel deg = 6')
plt.show()


# In[15]:


#plot of the MSE vs the hyperparamter C for polynomial kernel with degree = 2
x_values = np.linspace(1, 100, 10, endpoint=True)
mse_cvxopt = []
mse_sk = []
# mse_sklearn = []
for i in x_values:         
    mse1, mse2 = cross_val(X, y, 5, i, 0.1, 'poly', 1, 2)
    mse_cvxopt.append(mse1)
    mse_sk.append(mse2)

print(mse_cvxopt)
print(mse_sk)   
plt.scatter(x_values, mse_cvxopt, label ='cvxopt')
plt.scatter(x_values, mse_sk, label ='sklearn')

plt.xlabel(' Value of C')
plt.ylabel('MSE')
plt.legend()
plt.title('Polynomial kernel')
plt.show()
# plt.savefig('MEDV vs data.png')


# In[16]:


#plot of the MSE vs the hyperparamter gamma for polynomial kernel with degree = 2
x_values = np.linspace(1, 100, 10, endpoint=True)
mse_cvxopt = []
mse_sk = []
# mse_sklearn = []
for i in x_values:         
    mse1, mse2 = cross_val(X, y, 5, 1, 0.1, 'poly', i, 2)
    mse_cvxopt.append(mse1)
    mse_sk.append(mse2)

print(mse_cvxopt)
print(mse_sk)   
plt.scatter(x_values, mse_cvxopt, label ='cvxopt')
plt.scatter(x_values, mse_sk, label ='sklearn')

plt.xlabel(' Value of gamma')
plt.ylabel('MSE')
plt.legend()
plt.title('Polynomial kernel')
plt.show()
# plt.savefig('MEDV vs data.png')


# In[11]:


#finding the best hyperparameters for the polynomial kernel with degree = 2
min_error = 100
best_c = 0.1
best_gamma = 1
C_values = np.linspace(0.1, 2, 10)
gamma_values = np.linspace(0.1, 2, 10)
ac_mat = np.zeros([len(C_values), len(gamma_values)])
Xx, Yy = np.meshgrid(gamma_values, C_values)
for i in range(len(C_values)):
    for j in range(len(gamma_values)):        
        mse = cross_val_cvx(X, y, 5, C_values[i], 0.1, 'poly', gamma_values[j], 2)
        if mse < min_error:
            min_error = mse
            best_c = C_values[i]
            best_gamma = gamma_values[j]
        ac_mat[i, j] = mse
print('cvxopt mse : ', min_error, 'best_c : ', best_c, 'best_gamma : ', best_gamma)
mse_sk = cross_val_sk(X, y, 5, best_c, 1, 'poly', best_gamma, 2)
print('sklearn mse : ', mse_sk)
# print(np.sort(ans.support_))
# print(ans.support_.shape)

plt.figure(figsize=(9.5,6.5))
ax = plt.axes(projection='3d')
ax.plot_surface(Xx, Yy, ac_mat, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_xlabel('Value of Gamma')
ax.set_ylabel('Value of C')
ax.set_zlabel('MSE');
plt.show()
plt.contourf(Xx, Yy, ac_mat)
plt.colorbar()
plt.xlabel('Value of Gamma')
plt.ylabel('Value of C')
plt.title('Polynomial kernel')
plt.show()


# In[12]:


#plot of the MSE vs the degree for polynomial kernel
x_values = np.linspace(1, 5, 5, endpoint=True)
mse_cvxopt = []
mse_sk = []
# mse_sklearn = []
for i in x_values:         
    mse1, mse2 = cross_val(X, y, 5, 1, 0.1, 'poly', 1, i)
    mse_cvxopt.append(mse1)
    mse_sk.append(mse2)

print(mse_cvxopt)
print(mse_sk)   
plt.scatter(x_values, mse_cvxopt, label ='cvxopt')
plt.scatter(x_values, mse_sk, label ='sklearn')

plt.xlabel(' Degree ')
plt.ylabel('MSE')
plt.legend()
plt.title('Polynomial kernel')
plt.show()
# plt.savefig('MEDV vs data.png')


# # RH-SVR

# In[12]:


#formulation of the rh-SVR
def svr_cvxopt_rh(X_train, y_train, X_test, C, eps, kernel_type, gamma, degree):
    X = X_train
    y = y_train
    m,n = X.shape
    y = y.reshape(-1,1) * 1
#     print(y.shape)
    q1 = 2*eps*y
    K = kernel(kernel_type, X, X, gamma, degree)
    K1 = K + y@y.T
    p1 = np.hstack((K1, K1*-1))
    a1 = np.hstack((np.ones(m), np.zeros(m)))
    a2 = np.hstack((np.zeros(m), np.ones(m)))
    P = cvxopt_matrix(np.vstack((p1, p1*-1)))
    q = cvxopt_matrix(np.vstack((q1, q1*-1)))
    G = cvxopt_matrix(np.vstack((np.eye(2*m)*-1, np.eye(2*m))))
    h = cvxopt_matrix(np.hstack((np.zeros(2*m), np.ones(2*m) * C)))
    A = cvxopt_matrix(np.vstack((a1, a2)))
    # print(A.shape)
    b = cvxopt_matrix(np.ones(2))

    #Run solver
    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    variables = np.array(sol['x'])

    u = variables[:m]
    v = variables[m:2*m]
    delta = (u-v).T@y + 2*eps
    bias = (u-v).T@(K@(u+v))/(2*delta) + (u+v).T@y/2
    u1 = u/delta
    v1 = v/delta
    y_pred =  kernel(kernel_type, X, X_test, gamma, degree).T@(v1-u1) + bias
    #     print(y_pred)
    return y_pred


# In[13]:


# k -fold cross validation, returns the MSE for the given parameters for RH-SVR 
def cross_val_rh(X, y, num_folds, C, eps, kernel_type, gamma, degree):
    y = y.reshape(-1,1) * 1
    m, n = X.shape
    batch_size = m//num_folds

    mse1 = 0
    for i in range(1, num_folds+1):
        X_test = X[(i-1)*batch_size: i*batch_size, :]
        y_test = y[(i-1)*batch_size: i*batch_size:, ]
        if i == 1:
            X_train = X[i*batch_size : , :]
            y_train = y[i*batch_size :, :]
        else:
            X_train = X[: (i-1)*batch_size, :]
            np.vstack((X_train, X[i*batch_size : , :]))
            y_train = y[: (i-1)*batch_size, :]
            np.vstack((y_train, y[i*batch_size : , :]))
        
        y_pred = svr_cvxopt_rh(X_train, y_train, X_test, C, eps, kernel_type, gamma, degree)
        y_pred = y_pred[:, 0]
        y_test = y_test[:, 0]

        mse1 += (np.linalg.norm(y_pred - y_test) ** 2)/y_test.shape[0]
        
    mse1 = mse1/num_folds

    return mse1
    


# In[14]:


#plot of the predicted MEDV values by the RH-SVR vs the true values for the linear kernel
X_train = X[101: , :]
y_train = y[101: ]
X_test = X[:101, :]
y_test = y[:101]
# reg = svm.SVR(kernel = 'linear', degree = 2, C = 1, epsilon = 0.1).fit(X_train, y_train.ravel()) 
# y_sk = reg.predict(X_test)
# score1 = reg.score(X_test, y_test)
y_pred_rh = svr_cvxopt_rh(X_train, y_train, X_test, 1, 0.1, 'linear', 1, 2)
y_pred_eps = svr_cvxopt(X_train, y_train, X_test, 1, 0.1, 'linear', 1, 2)

y_pred_eps = y_pred_eps[:, 0]
y_pred_rh = y_pred_rh[:, 0]
print((np.linalg.norm(y_pred_eps - y_test) ** 2)/y_test.shape[0])
print((np.linalg.norm(y_pred_rh - y_test) ** 2)/y_test.shape[0])
# print(np.max(y_sk-y_test))

Xx = np.array(list(range(0, y_test.shape[0]))).reshape(y_test.shape[0], 1)
plt.plot(Xx, y_test, label ='True')
plt.plot(Xx, y_pred_rh, label ='Predicted RH-svr')

plt.plot(Xx, y_pred_eps, label ='Predicted eps-SVR')
plt.xlabel('Data Points')
plt.ylabel('MEDV')
plt.legend()
plt.title('linear Kernel')
plt.show()


# In[17]:


#plot of the MSE vs the hyperparamter C 
C_values = np.linspace(1, 10, 10, endpoint=True)
mse_cvxopt = []
mse_cvxopt_rh = []
mse_sk = []
# mse_sklearn = []
for i in C_values:         
    mse1 = cross_val_cvx(X, y, 5, i, 0.1, 'linear', 1, 1)
    mse2 = cross_val_rh(X, y, 5, i, 0.1, 'linear', 1, 1)
    mse3 = cross_val_sk(X, y, 5, i, 0.1, 'linear', 1, 1)
    mse_cvxopt.append(mse1)
    mse_cvxopt_rh.append(mse2)
    mse_sk.append(mse3)

print(mse_cvxopt)
print(mse_cvxopt_rh)
plt.scatter(C_values, mse_cvxopt, label ='eps-SVR')
plt.scatter(C_values, mse_cvxopt_rh, label ='rh_SVR')
# plt.scatter(C_values, mse_sk, label ='sklearn')

plt.xlabel(' Value of C')
plt.ylabel('MSE')
plt.legend()
plt.title('Linear kernel')
plt.show()
# plt.savefig('MEDV vs data.png')


# In[21]:


#plot of the MSE vs the hyperparamter C 
x_values = np.linspace(0.01, 0.1, 10, endpoint=True)
mse_cvxopt = []
mse_cvxopt_rh = []
mse_sk = []
# mse_sklearn = []
for i in x_values:         
    mse1 = cross_val_cvx(X, y, 5, 1, i, 'linear', 1, 1)
    mse2 = cross_val_rh(X, y, 5, 1, i, 'linear', 1, 1)
#     mse3 = cross_val_sk(X, y, 5, 1, 0.1, 'linear', 1, 1)
    mse_cvxopt.append(mse1)
    mse_cvxopt_rh.append(mse2)
    mse_sk.append(mse3)

print(mse_cvxopt)
print(mse_cvxopt_rh)
plt.scatter(x_values, mse_cvxopt, label ='eps-SVR')
plt.scatter(x_values, mse_cvxopt_rh, label ='rh_SVR')
# plt.scatter(C_values, mse_sk, label ='sklearn')

plt.xlabel(' Value of eps')
plt.ylabel('MSE')
plt.legend()
plt.title('Linear kernel')
plt.show()
# plt.savefig('MEDV vs data.png')


# In[27]:


ktype = ['linear', 'rbf', 'poly']
mse_cvxopt = []
mse_cvxopt_rh = []
# mse_sklearn = []
for i in ktype:         
    mse1 = cross_val_cvx(X, y, 5, 1, 0.1, i, 1, 2)
    mse2 = cross_val_rh(X, y, 5, 1, 0.1, i, 1, 2)
    mse_cvxopt.append(mse1)
    mse_cvxopt_rh.append(mse2)

print(mse_cvxopt)
print(mse_cvxopt_rh)
plt.scatter(ktype, mse_cvxopt, label ='eps-SVR')
plt.scatter(ktype, mse_cvxopt_rh, label ='RH_SVR')

plt.xlabel(' Kernel')
plt.ylabel('MSE')
plt.legend()
plt.title('Comparision of kernels')
plt.show()
# plt.savefig('MEDV vs data.png')


# In[ ]:




