import numpy as np
from collections import Counter




class _BaseKnn:
    def __init__(self,k):
        self.k = k

    
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y

    
    def k_neighbours(self,X):
        distance = [np.linalg.norm(X-q) for q in self.X_train]
        n_neighbours_ind = np.argsort(distance)[:self.k]
        k_neighbours_label = [self.y_train[i] for i in n_neighbours_ind]

        return k_neighbours_label
    
class KNNClassifier:
    def __init__(self,k):
        self.k = k
        self.knn = _BaseKnn(self.k)

    def fit(self,X,y):
        self.knn.fit(X,y)

    def predict(self,X):
        predicted_label = np.array([self._predict(x) for x in X])
        return predicted_label

    def _predict(self,X):
        n_neighbours = self.knn.k_neighbours(X)
        return Counter(n_neighbours).most_common(1)[0][0]
    
class KNNRegressor:
    def __init__(self,k):
        self.k = k
        self.knn = _BaseKnn(self.k)

    def fit(self,X,y):
        self.knn.fit(X,y)

    def predict(self,X):
        predicted_label = np.array([self._predict(x) for x in X])
        return predicted_label

    def _predict(self,X):
        n_neighbours = self.knn.k_neighbours(X)
        return np.mean(n_neighbours)



        



