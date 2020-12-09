# 실습
# 1. 상단 모델에 그리드서치 또는 랜덤서치 적용
# 최적의 R2값과 피쳐임포턴스 구할것

# 2. 위 쓰레드값으로 SelectFromModel을 구해서 
# 최적의 피쳐갯수를 구할것

# 3. 위 피쳐 갯수로 데이터(피쳐)를 수정(삭제)해서
# 그래드서치 또는 랜덤서치 적용
# 최적의 R2값을 구할 것

# 1번값과 2번값을 비교해볼것

import numpy as np
from sklearn.datasets import load_boston
from xgboost import XGBClassifier, XGBRegressor, plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV


boston = load_boston()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, train_size=0.7, random_state=66,shuffle=True)


parameters = [
    {"n_estimators":[100, 200, 300], "learning_rate":[0.1, 0.3, 0.001, 0.01], "max_depth":[4,5,6]},
    {"n_estimators":[90, 100, 110], "learning_rate":[0.1, 0.001, 0.01], "max_depth":[4,5,6]},
    {"n_estimators":[100, 200, 300], "learning_rate":[0.1, 0.3, 0.001, 0.001], "max_depth":[4,5,6],
      "colsample_bytree":[0.6, 0.9, 1], "colsample_bylevel":[0.6, 0.7, 0.9]}
]

model = RandomizedSearchCV(XGBRegressor(), parameters, cv=5,verbose=2)

model.fit(x_train,y_train)
acc = model.score(x_test,y_test)
print("acc1 : ",acc)

print("최적의 estimator : ",model.best_estimator_)
print("최적의 params    : ",model.best_params_)

y_predict = model.predict(x_test)
print("최종 정답률     : ",r2_score(y_test,y_predict))


model = model.best_estimator_

thresholds = np.sort(model.feature_importances_)
print(thresholds)

n  = 0
r2 = 0

import time
start1 = time.time()

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)
    selection_model = XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train,y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)
    # if score*100.0 > r2:
    #     n = select_x_train.shape[1]
    #     r2 = score*100.0
    #     L_selection = selection
    #     print("Thresh=%.3f, n=%d, R2: %.2f%%"%(thresh,select_x_train.shape[1],score*100.0))
    
start2 = time.time()

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)
    selection_model = XGBRegressor(n_jobs=1)
    selection_model.fit(select_x_train,y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)
    # if score*100.0 > r2:
    #     n = select_x_train.shape[1]
    #     r2 = score*100.0
    #     L_selection = selection
    #     print("Thresh=%.3f, n=%d, R2: %.2f%%"%(thresh,select_x_train.shape[1],score*100.0))
end2 = time.time()
end = start2 - start1
print("start1 : ",end)
print("start2 ",end2-start2) 

'''
x_train = L_selection.transform(x_train)
x_test = L_selection.transform(x_test)

model = RandomizedSearchCV(XGBRegressor(), parameters, cv=5,verbose=2)

model.fit(x_train,y_train)
acc = model.score(x_test,y_test)
print("acc1 : ",acc)

print("최적의 estimator : ",model.best_estimator_)
print("최적의 params    : ",model.best_params_)

y_predict = model.predict(x_test)
print("최종 정답률     : ",r2_score(y_test,y_predict))

model = model.best_estimator_

thresholds = np.sort(model.feature_importances_)
print(thresholds)

# 1번
# 최적의 estimator :  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
#              colsample_bynode=1, colsample_bytree=0.9, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.3, max_delta_step=0, max_depth=4,
#              min_child_weight=1, missing=nan, monotone_constraints='()',
#              n_estimators=200, n_jobs=0, num_parallel_tree=1, random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#              tree_method='exact', validate_parameters=1, verbosity=None)
# 최적의 params    :  {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.3, 'colsample_bytree': 0.9, 'colsample_bylevel': 0.6}
# 최종 정답률     :  0.8966062674547045
# [0.00415122 0.00711221 0.00941916 0.01142439 0.01667747 0.01917566
#  0.02122034 0.02278667 0.04694101 0.05007362 0.14996621 0.22987813
#  0.41117394]


# 2번
# 최적의 estimator :  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
#              colsample_bynode=1, colsample_bytree=0.6, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.1, max_delta_step=0, max_depth=5,
#              min_child_weight=1, missing=nan, monotone_constraints='()',
#              n_estimators=200, n_jobs=0, num_parallel_tree=1, random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#              tree_method='exact', validate_parameters=1, verbosity=None)
# 최적의 params    :  {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1, 'colsample_bytree': 0.6, 'colsample_bylevel': 0.6}
# 최종 정답률     :  0.9103265848524855
# [0.00561222 0.0121988  0.02149129 0.02638417 0.02788421 0.05868933
#  0.06518003 0.0655624  0.0738856  0.11177695 0.23410691 0.29722813]

'''






















