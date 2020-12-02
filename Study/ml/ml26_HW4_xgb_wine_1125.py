# pipe라인 까지 구성할 것
# csv 파일 사용할 것
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from xgboost import XGBClassifier, XGBRegressor, plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline, make_pipeline


# 1. data
dataset = pd.read_csv("./data/csv/winequality-white.csv",header=0,index_col=None,sep=";")
print(dataset.describe())

datasets = pd.DataFrame(dataset)
x = datasets.iloc[:,:11]
y = datasets.iloc[:,11]

x = np.array(x)
y = np.array(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=66,shuffle=True)


parameters = [
    {"svm__n_estimators":[100, 200, 300], "svm__learning_rate":[0.1, 0.3, 0.001, 0.01], 
    "svm__max_depth":[4,5,6]},
    {"svm__n_estimators":[90, 100, 110], "svm__learning_rate":[0.1, 0.001, 0.01], 
    "svm__max_depth":[4,5,6]},
    {"svm__n_estimators":[100, 200, 300], "svm__learning_rate":[0.1, 0.3, 0.001, 0.001], 
    "svm__max_depth":[4,5,6],
      "svm__colsample_bytree":[0.6, 0.9, 1], "svm__colsample_bylevel":[0.6, 0.7, 0.9]}
]


pipe = Pipeline([("scaler",MinMaxScaler()), ("svm",XGBClassifier())])


model = RandomizedSearchCV(pipe,parameters,cv=5, verbose=2)


model.fit(x_train,y_train)
acc = model.score(x_test,y_test)
print("acc1 : ",acc)

print("최적의 estimator : ",model.best_estimator_)
print("최적의 params    : ",model.best_params_)

y_predict = model.predict(x_test)
print("최종 정답률     : ",accuracy_score(y_test,y_predict))

# 최적의 estimator :  Pipeline(steps=[('scaler', MinMaxScaler()),
#                 ('svm',
#                  XGBClassifier(base_score=0.5, booster='gbtree',
#                                colsample_bylevel=0.7, colsample_bynode=1,
#                                colsample_bytree=0.9, gamma=0, gpu_id=-1,
#                                importance_type='gain',
#                                interaction_constraints='', learning_rate=0.3,
#                                max_delta_step=0, max_depth=6,
#                                min_child_weight=1, missing=nan,
#                                monotone_constraints='()', n_estimators=200,
#                                n_jobs=0, num_parallel_tree=1,
#                                objective='multi:softprob', random_state=0,
#                                reg_alpha=0, reg_lambda=1, scale_pos_weight=None,
#                                subsample=1, tree_method='exact',
#                                validate_parameters=1, verbosity=None))])
# 최적의 params    :  {'svm__n_estimators': 200, 'svm__max_depth': 6, 'svm__learning_rate': 0.3, 'svm__colsample_bytree': 0.9, 'svm__colsample_bylevel': 0.7}
# 최종 정답률     :  0.6768707482993197