import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV

from sklearn.datasets import load_iris

dataset = load_iris()
x = dataset.data
y = dataset.target

# x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1207,shuffle=True)


# 2. 모델
model = XGBClassifier(n_estimators=2000, learning_rate=0.1)
# model = XGBRegressor(learning_rate=0.1)

# 3. 훈련
model.fit(x_train,y_train,verbose=True, eval_metric=["merror","mlogloss"], 
          eval_set=[(x_train,y_train),(x_test,y_test)])
# [1999]  validation_0-merror:0.00000     validation_0-mlogloss:0.01554   
#         validation_1-merror:0.06667     validation_1-mlogloss:0.35598

# rmse, mae, logloss, error, auc

results = model.evals_result()
print("evals_result : ",results["validation_0"])


y_pred = model.predict(x_test)

acc = accuracy_score(y_test,y_pred)
print("acc : ",acc)
# acc :  0.9333333333333333






























