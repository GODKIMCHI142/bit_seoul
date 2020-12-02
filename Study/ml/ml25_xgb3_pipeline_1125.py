# 과적합 방지
# 1. 훈련데이터량을 늘린다.
# 2. 피쳐수를 줄인다.
# 3. regularization

from sklearn.datasets import load_boston
from xgboost import XGBClassifier, XGBRegressor, plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline, make_pipeline
boston = load_boston()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, train_size=0.7, random_state=66,shuffle=True)


parameters = [
    {"svm__n_estimators":[100, 200, 300], "svm__learning_rate":[0.1, 0.3, 0.001, 0.01], 
    "svm__max_depth":[4,5,6]},
    {"svm__n_estimators":[90, 100, 110], "svm__learning_rate":[0.1, 0.001, 0.01], 
    "svm__max_depth":[4,5,6]},
    {"svm__n_estimators":[100, 200, 300], "svm__learning_rate":[0.1, 0.3, 0.001, 0.001], 
    "svm__max_depth":[4,5,6],
      "svm__colsample_bytree":[0.6, 0.9, 1], "svm__colsample_bylevel":[0.6, 0.7, 0.9]}
]


pipe = Pipeline([("scaler",MinMaxScaler()), ("svm",XGBRegressor())])


model = RandomizedSearchCV(pipe,parameters,cv=5, verbose=2)


model.fit(x_train,y_train)
acc = model.score(x_test,y_test)
print("acc1 : ",acc)

print("최적의 estimator : ",model.best_estimator_)
print("최적의 params    : ",model.best_params_)

y_predict = model.predict(x_test)
print("최종 정답률     : ",r2_score(y_test,y_predict))


# 최적의 estimator :  Pipeline(steps=[('scaler', MinMaxScaler()),
#                 ('svm',
#                  XGBRegressor(base_score=0.5, booster='gbtree',
#                               colsample_bylevel=0.7, colsample_bynode=1,
#                               colsample_bytree=0.6, gamma=0, gpu_id=-1,
#                               importance_type='gain',
#                               interaction_constraints='', learning_rate=0.1,
#                               max_delta_step=0, max_depth=5, min_child_weight=1,
#                               missing=nan, monotone_constraints='()',
#                               n_estimators=100, n_jobs=0, num_parallel_tree=1,
#                               random_state=0, reg_alpha=0, reg_lambda=1,
#                               scale_pos_weight=1, subsample=1,
#                               tree_method='exact', validate_parameters=1,
#                               verbosity=None))])
# 최적의 params    :  {'svm__n_estimators': 100, 'svm__max_depth': 5, 'svm__learning_rate': 0.1, 'svm__colsample_bytree': 0.6, 'svm__colsample_bylevel': 0.7}
# 최종 정답률     :  0.9111833384162625








