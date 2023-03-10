import numpy as np

#let X be shape : (training examples, features)
#let y be shape : (training_examples, 1)
#output w be shape: (features, 1)




def linear_regression_normal_equation(X, y):
    ones = np.ones((X.shape[0], 1))
    X = np.append(ones, X, axis=1)
    W = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, y))
    return W


if __name__ == "__main__":
    # Run a small test example: y = 5x (approximately)
    m, n = 500, 1
    X = np.random.rand(m, n)
    y = 5 * X + np.random.randn(m, n) * 0.1
    W = linear_regression_normal_equation(X, y)
    print(W)