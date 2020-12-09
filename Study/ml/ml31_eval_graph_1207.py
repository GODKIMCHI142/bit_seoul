import numpy as np
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV

from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target

# x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1207,shuffle=True)


# 2. 모델
model = XGBRegressor(n_estimators=1000, learning_rate=0.1)
# model = XGBRegressor(learning_rate=0.1)

# 3. 훈련

model.fit(x_train,y_train,verbose=True, eval_metric=["logloss","rmse"], 
          eval_set=[(x_train,y_train),(x_test,y_test)])
# Stopping. Best iteration:
# [167]   validation_0-rmse:0.19338       validation_1-rmse:3.09635

# rmse, mae, logloss, error, auc

results = model.evals_result()
print("evals_result : ",results["validation_0"])


y_pred = model.predict(x_test)

r2 = r2_score(y_test,y_pred)
print("r2 : ",r2)
# r2 :  0.8803134022164779

# 시각화
import matplotlib.pyplot as plt

train = results["validation_0"]["logloss"]
test  = results["validation_1"]["logloss"]
epochs = len(results["validation_0"]["logloss"])
x_axis = range(0,epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, train, label="Train")
ax.plot(x_axis, test,  label="Test")
ax.legend()
plt.ylabel("Log Loss")
plt.title("XGBoost Log Loss")
plt.show()

train = results["validation_0"]["rmse"]
test  = results["validation_1"]["rmse"]
epochs = len(results["validation_0"]["rmse"])
x_axis = range(0,epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, train, label="Train")
ax.plot(x_axis, test,  label="Test")
ax.legend()
plt.ylabel("Rmse")
plt.title("XGBoost Rmse")
plt.show()






















