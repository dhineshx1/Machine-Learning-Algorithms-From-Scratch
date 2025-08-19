import numpy as np

class LinearRegressor:
    def __init__(self, lr=0.001, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_samples, self.n_features = self.X.shape
        self.weights = np.zeros(self.n_features)
        self.bias = 0

        # Perform gradient descent
        for _ in range(self.n_iter):
            y_pred = np.dot(self.X, self.weights) + self.bias

            # Compute gradients
            dw = (1 / self.n_samples) * np.dot(self.X.T, (y_pred - self.y))
            db = (1 / self.n_samples) * np.sum(y_pred - self.y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


    
        






        