import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from neighbors import KNNClassifier,KNNRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from linear_regression import LinearRegressor
from logistic_regressor import LogisticRegressor

def test_KNNClassifier():
    iris = datasets.load_iris()

    X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,random_state=42,test_size=0.8)

    model = KNNClassifier(4)

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    print(confusion_matrix(y_pred,y_test))
    acc =np.sum(y_pred==y_test) / len(y_pred)
    print(acc)

def test_KNNRegressor():
    diabetes  = datasets.load_diabetes()
    X_train,X_test,y_train,y_test = train_test_split(diabetes.data,diabetes.target,random_state=42,test_size=0.8)

    model = KNNRegressor(5)

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("KNN Regressor Evaluation:")
    print(f"  MSE  : {mse:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  MAE  : {mae:.2f}")
    print(f"  R sqr   : {r2:.2f}")

def test_linear_regression():
    diabetes  = datasets.load_diabetes()
    X_train,X_test,y_train,y_test = train_test_split(diabetes.data,diabetes.target,random_state=42,test_size=0.8)

    model = LinearRegressor(lr=0.1,n_iter=2000)

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Linear Regressor Evaluation:")
    print(f"  MSE  : {mse:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  MAE  : {mae:.2f}")
    print(f"  R sqr   : {r2:.2f}")


def test_logistic_regressor():
    bc = datasets.load_breast_cancer()

    X_train,X_test,y_train,y_test = train_test_split(bc.data,bc.target,random_state=42,test_size=0.8)

    model = LogisticRegressor(lr=0.01, n_iter=2000)

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    print(confusion_matrix(y_pred,y_test))
    acc =np.sum(y_pred==y_test) / len(y_pred)
    
    print(acc)

test_KNNClassifier()
test_KNNRegressor()
test_linear_regression()
test_logistic_regressor()



