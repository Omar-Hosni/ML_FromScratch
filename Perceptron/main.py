#inputs + weights -> net input function (w1x1 + w2x2 ... + wnxn) + activation function (sigmoid)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iters):

            for idx, x_i in enumerate(X):

                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                # Perceptron update rule
                update = self.lr * (y_[idx] - y_predicted)

                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, 0)  




def accuracy(y_pred, y_true):
    return np.sum(y_true == y_pred) / len(y_true)


if __name__ == '__main__':
    X, y = datasets.make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=2, random_state=2)

    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    perceptron = Perceptron(learning_rate=0.01, n_iters=1000)
    perceptron.fit(X, y)
    y_pred = perceptron.predict(X_test)

    print('Perceptron accuracy', accuracy(y_pred, y_test))
