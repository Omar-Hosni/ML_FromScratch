import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iter=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iter = n_iter
        self.w = None
        self.b = None
    
    def fit(self, X,y):
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w)-self.b) >= 1

                if condition:
                    self.w -= self.lr * (2*self.lambda_param*self.w)
                else:
                    self.w -= self.lr * (2*self.lambda_param*self.w - np.dot(y_[idx],x_i))
                    self.b -= self.lr * y_[idx]


    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)




from sklearn import datasets

if __name__ == '__main__':

    X,y  = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
    y = np.where(y==0,-1,1)

    svm = SVM()
    svm.fit(X,y)
    y_pred = svm.predict(X)

    print(y_pred)
