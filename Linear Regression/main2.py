import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt 

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)

fig = plt.figure(figsize=(8,6))
#plt.scatter(X[:, 0],y,color="b",marker="o",s=30)
#plt.show()



class LinearRegression:

    def __init__(self, lr=0.001, n_iter=1000):
        self.lr = lr
        self.n_iter=  n_iter
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            y_predicted = np.dot(X, self.weights)+ self.bias 

            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted-y)

            self.weights -= self.lr*dw
            self.bias -= self.lr *db

    def predict(self, X):
        y_approximated = np.dot(X, self.weights) + self.bias
        return y_approximated


#cost function
def mse(y_true,y_predicted):
    np.mean((y_true-y_predicted)**2)

if __name__ == '__main__':
    regressor = LinearRegression(lr=0.001)
    regressor.fit(X_train, y_train)
    predicted = regressor.predict(X_test)

    mse_value = mse(y_test, predicted)

    print(mse_value)


    