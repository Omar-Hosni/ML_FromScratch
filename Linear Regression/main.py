#m = #traing examples
#n = #number of features
#y is R^(l*m)
#X is R^(n*m)
#w si R^(n*1)
#y is the label and yhat is the prediction of that label
#alpha/learning rate is the steps taken in the gradient descent to reach the minimum value
#the minimum value is the most minimum distance between the linear regression and the points

'''
steps of gradient descent 

loop thorugh 1000 times{
    y' = matrix_multiplication(X,theta)
    cost/loss = 1/2m * sum[(y,y')^2]
    dtheta = 1/m * matrix_multiplication(X.T, y-y')
    theta = theta - alpha*dtheta
}

# y' is the yhal which is the predict point
# dtheta is the derivative of the cost function
# you need the derivative of the cost/loss function to do the gradient descent
'''



import numpy as np


class LinearRegression():
    def __init__(self):
        
        self.learning_rate = 0.01 #alpha/learning rate
        self.total_iterations = 10000

    def yhat(self, X, w): #yhat = w1x1 + w2x2 + w3x2 ....
        return np.dot(w.T, X)
    
    #loss/cost function
    def loss(self, yhat, y):
        L = 1/self.m * np.sum(np.power(yhat-y, 2))
        return L

    def gradient_descent(self, w, X, y, yhat):
        dLdW = 2/self.m * np.dot(X ,(yhat-y).T)
        w = w - self.learning_rate * dLdW
        return w

    def main(self, X,y):
        x1 = np.ones((1,X.shape[1]))
        X = np.append(X, x1, axis=0)

        self.m = X.shape[1]
        self.n = X.shape[0]

        w = np.zeros((self.n, 1))

        for it in range(self.total_iterations+1):
            yhat = self.yhat(X,w)
            loss = self.loss(yhat, y)

            if it % 2000 == 0:
                print(f'Cost at iteration {it} is {loss}')

            w = self.gradient_descent(w, X, y, yhat)



if __name__ == '__main__':
    X = np.random.rand(1,500) #y = w1 + w2x2 + ...
    y = 3*X+np.random.randn(1,500) * 0.1

    regression = LinearRegression()
    w = regression.main(X,y)

    