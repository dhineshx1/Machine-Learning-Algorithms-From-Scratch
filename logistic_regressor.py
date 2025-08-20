import numpy as np


class LogisticRegressor:
    """
    This Logistic regressor only works with Binary Classification
    
    """
    def __init__(self,lr=0.001,n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bais = None

    def fit(self,X,y):
        self.X = X
        self.y = y
        no_of_samples, no_of_features = X.shape
        self.weights = np.zeros(no_of_features)
        self.bais = 0

        for _ in range(self.n_iter):
            model = np.dot(self.X,self.weights) + self.bais
            y_pred = self._sigmoid(model)

            dw = 1/no_of_samples * np.dot(self.X.T,(y_pred-self.y))
            db = 1/no_of_samples * np.sum(y_pred-self.y)

            self.weights -= self.lr * dw
            self.bais -= self.lr * db


    def _sigmoid(self,value):
        return 1/ (1 + np.exp(-value))
    
    def predict(self,X):
        model = np.dot(X,self.weights) + self.bais
        y_pred = self._sigmoid(model)
        y_pred_label = [ 1 if y > 0.5 else 0 for y in y_pred]
        return y_pred_label

        