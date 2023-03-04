#approximation f(w,b) = wx+b
# y predict = h(x) = 1/e^-wx+b
#sigmoid - s(x) = 1/1+e^-x
#cost function J(w,b) = J(theta) = 1/n sum(ylog(h(xi)) + (1- y)log(1-h(x))
#updates -> w=w-alpha*dw , b=b-alpha*db

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class LogisticRegression:

    def __init__(self, lr=0.001, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self,X,y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        #gradient descent
        for _ in range(self.n_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            dw = 1/n_features * np.dot(X.T,(y_predicted-y))
            db = 1/n_samples * np.sum(y_predicted-y)
            
            self.weights -= self.lr*dw
            self.bias -= self.lr*db


    def predict(self, X,):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)

        y_predicted_cls = [1 if y_pred > 0.5 else 0 for y_pred in y_predicted]
        return y_predicted_cls

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))


def testAccuracy(y_true, y_pred):
    accuracy = np.sum(y_test == y_true) / len(y_true)
    return accuracy

if __name__ == '__main__':
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    
    regressor = LogisticRegression(lr=0.0001, n_iter=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    print('Accuracy= ',testAccuracy(y_test, predictions))
