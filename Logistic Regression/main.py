#Logistic Regreggion-> J(Θ) = -1/m (mΣi=1 * y^(i) * logh(Θ)(x^i) + (1-y^i) * log(1-h(Θ)(x^i)))
#sigmoid/hypothesis function-> h(x^i) = 1/1+e^-(wTx+b)
#cost-> to find the theta parameters or the weights 
#Logistic Regreggion -> J(Θ) = 1/m * mΣi=1(cost(hΘ(x^i,y^i)))

#cost function if y=0 -> -log(1-hΘ(x))
#cost function if y=1 -> -log(hΘ(x))
#cost(hΘ|x, y)= -ylog(hΘ(x)) - (1-y)*log(1-y*hΘ(x))


import numpy as np
from sklearn.datasets import make_blobs

class LogistiRegression():
    def __init__(self, X, learning_rate=0.1, num_iters=10000):
        self.lr = learning_rate
        self.num_iters = num_iters
        
        #m for training examples, n for features
        self.m, self.n = X.shape
    
    def sigmoid(self, z):
        #z = wTx+b, this is what defines the linear classifier
        return 1 / (1+np.exp(-z)) 

    def train(self, X, y):
        self.weights = np.zeros((self.n, 1))
        self.bias = 0

        for it in range(self.num_iters+1):
            #calculate hypothesis
            y_pred = self.sigmoid(np.dot(X, self.weights)+self.bias)
            cost = -1/self.m * np.sum(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))

            #backprop
            dw = 1/self.m * np.dot(X.T, (y_pred-y)) #cost in perspect of w in vectorized way 
            db = 1/self.m * np.sum(y_pred-y)

            self.weights -= self.lr * dw #gradient descent
            self.bias -= self.lr * db

            if it % 1000 == 0:
                print(f'cost after iteration {it}: {cost}')

        return self.weights, self.bias

    def predict(self, X):
        y_pred = self.sigmoid(np.dot(X, self.weights)+self.bias)
        y_pred_labels = y_pred > 0.5

        return y_pred_labels




if __name__ == '__main__':
    np.random.seed(1)
    X,y = make_blobs(n_samples=1000, centers=2)
    y = y[:, np.newaxis] #(1000,) = #(1000,1) changing dimensions

    logreg = LogistiRegression(X)
    w,b =logreg.train(X,y)

    y_pred = logreg.predict(X)

    print(f'Accuracy: {np.sum(y==y_pred)/X.shape[0]}')
    

