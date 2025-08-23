import numpy as np


def mse(y_pred,y_true):
    error = np.mean((y_true - y_pred) ** 2)
    return error

def accuracy(y_pred,y_true):
    acc = np.mean(y_pred==y_true)
    return acc