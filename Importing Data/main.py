import csv 
import numpy as np
import pandas as pd

FILE_NAME = 'spambase.data'

#data = np.loadtxt(FILE_NAME, delimiter=',')
data = np.genfromtxt(FILE_NAME, delimiter=',')




df = pd.read_csv(FILE_NAME,header=None, delimiter=',')
data = df.to_numpy()

print(data.shape)

n_samples, n_features = data.shape 

n_features -= 1
X = data[:,0:n_features]
y = data[:,n_features]

print(X.shape, y.shape)