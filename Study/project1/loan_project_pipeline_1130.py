import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import LinearSVC, SVC

# 1. 데이터 
x = np.load("./data/loan_x.npy")
y = np.load("./data/loan_y.npy")

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=30,shuffle=True)


pipe = make_pipeline(MaxAbsScaler(), XGBClassifier())

pipe.fit(x_train,y_train)

acc = pipe.score(x_test,y_test)

print("acc : ",acc)
# acc :  0.7822264159035834 -> scaler 모두 똑같음