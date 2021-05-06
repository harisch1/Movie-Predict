import numpy as np
class LinearRegression:

    def __init__(self, lr=0.001,n = 20000):
        self.lr = lr
        self.weights = None
        self.bias = None
        self.n = n
    
    def fit (self, X, y)
        #initializing parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for i in range(self.n):
            # y_hat = (theta1*x1 + ... + thetai*xi) + theta0
            y_pred = np.dot(X,self.weights) + bias
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
