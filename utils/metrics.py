import numpy as np

def mse(y_pred,y_true):
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_pred,y_true):
    return np.sqrt(mse(y_pred,y_true))

def mae(y_pred,y_true):
    return np.mean(np.abs(y_true - y_pred))

def accuracy(y_pred,y_true):
    return np.mean(y_pred == y_true)

def _cm_metrics(y_pred,y_true):
    TP = np.sum((y_pred==1) & (y_true==1))
    TN = np.sum((y_pred==0) & (y_true==0))
    FP = np.sum((y_pred==1) & (y_true==0))
    FN = np.sum((y_pred==0) & (y_true==1))
    return TP, TN, FP, FN

def precision(y_pred,y_true):
    TP = np.sum((y_true==1) & (y_pred==1))
    FP = np.sum((y_true==0) & (y_pred==1))
    return TP/(TP+FP) if (TP+FP) > 0 else 0

def recall(y_pred,y_true):
    TP = np.sum((y_pred==1) & (y_true==1))
    FN = np.sum((y_pred==0) & (y_true==1))
    return TP/(TP+FN) if (TP+FN) > 0 else 0
