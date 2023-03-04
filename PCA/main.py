import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        #mean
        self.mean = np.mean(X, axis=0)
        X = X-self.mean

        #covariance
        # row = sample, columns = feature
        cov = np.cov(X.T)

        #eigenvectors, eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        

        #sort eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1] #sort list from start to end in descending
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]



        #store first n eigenvectors
        self.components = eigenvectors[0:self.n_components]


    def transform(self, X):
        X = X-self.mean
        return np.dot(X, self.components.T)

    




from sklearn import datasets
import matplotlib.pyplot as plt

def accurace(y_true,y_pred):
    return np.sum(y_true==y_pred)/len(y_true)

if __name__ == '__main__':
    data = datasets.load_iris()
    X = data.data
    y = data.target

    pca = PCA(2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print('Shape of X ', X.shape)
    print('Shape of transformed X ', X_projected.shape)

    x1 = X_projected[:,0]
    x2 = X_projected[:,1]

    plt.scatter(x1,x2, c=y, edgecolor='none', alpha=0.8, cmap=plt.cm.get_cmap('viridis',3))

    plt.xlabel('Principal Component1')
    plt.ylabel('Principal Component2')

    plt.colorbar()
    plt.show()