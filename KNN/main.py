#np.argsort(np.array([1,2,3,4]) = sorts numbers based on who's the closest, number is represented by its index
#np.bincount(np.array([1,2,3,4]) = counts occurrency of number in each index, first index represents number 1
#np.argmax(np.array[1,2,3]) = biggest argument of them


import numpy as np

class KNearestNeighbor():
    def __init__(self,k):
        self.k=k
        self.eps = 1e-8
    
    def train(self,X,y):
        self.X_train = X
        self.y_train = y
        
    def predict_lables(self, distances):
        num_test = distances.shape[0]
        y_pred = np.zeros(num_test)
            
        for i in range(num_test):
            y_indices = np.argsort(distances[i,:])
            k_closest_classes = self.y_train[y_indices[:self.k]].astype(int)
            y_pred[i] = np.argmax(np.bincount(k_closest_classes))
                
        return y_pred


    def compute_distance_and_vectorize(self, X_test):
        #(X_Test - X_Train)^2 = X_Test**2 - 2*X_Test*X_Train + X_Train**2
        #most efficient way
        X_test_squared = np.sum(X_test**2, keepdims=True, axis=1)
        X_train_squared = np.sum(self.X_train**2,keepdims=True, axis=1)
        X_test_train_dot = np.dot(X_test, self.X_train.T)
        
        #self.eps to make it more numerically stable in case distance close to zero
        distances = np.sqrt(self.eps + X_test_squared - 2*X_test_train_dot + X_train_squared.T) #.T to transpose so we can add

        return distances
        
    
    def compute_distance_one_loop(self, X_test):
        #more efficient
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        distances = np.zeros((num_test, num_train))

        for i in range(num_test):
            distances[i,:] = np.sqrt(np.sum((self.X_train - X_test[i,:])**2,axis=1))

        
        return distances
    
    def compute_distance_two_loops(self, X_test):
        #Naive, inefficient way
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        distances = np.zeros((num_test, num_train))
        
        for i in range(num_test):
            for j in range(num_train):
                distances[i,j] = np.sqrt(np.sum((X_test[i,:] - self.X_train[j,:])**2)) #distance equation
        
        return distances
    
    
    def predict(self,X_test, num_loops=2):
        if num_loops == 2:    
            distances = self.compute_distance(X_test)
        elif num_loops == 1:
            distances = self.compute_distance_one_loop(X_test)  
        else:
            distances = self.compute_distance_and_vectorize(X_test)
         
        return self.predict_lables(distances)
    
    
    
    
 
    
    
if __name__ == '__main__':



    X = np.loadtxt("data.txt", delimiter=",")
    y = np.loadtxt("targets.txt")
    KNN = KNearestNeighbor(k=3)
    KNN.train(X,y)
    y_pred =KNN.predict(X, num_loops=0)
    
    print(f'Accuracy: {sum(y_pred==y)/y.shape[0]}')
    
    
    