import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from xgboost import XGBClassifier, XGBRegressor, plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import LinearSVC, SVC

# 1. 데이터 
x         = np.load("./data/loan_x.npy")
x_predict = np.load("./data/loan_x_predict.npy")
y         = np.load("./data/loan_y.npy")

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=30,shuffle=True)

model = XGBClassifier()

model.fit(x_train,y_train)

acc = model.score(x_test,y_test)
print("acc : ",acc)

x_FN = ["disbursed_amount","asset_cost", "ltv", "Date.of.Birth","Employment.Type", "MobileNo_Avl_Flag",
"Aadhar_flag", "PAN_flag", "VoterID_flag", "Passport_flag", "PERFORM_CNS.SCORE", "PRI.NO.OF.ACCTS",
"PRI.ACTIVE.ACCTS", "PRI.OVERDUE.ACCTS", "PRI.CURRENT.BALANCE", "PRI.SANCTIONED.AMOUNT", 
"PRI.DISBURSED.AMOUNT", "SEC.NO.OF.ACCTS", "SEC.ACTIVE.ACCTS", "SEC.OVERDUE.ACCTS", "SEC.CURRENT.BALANCE",
"SEC.SANCTIONED.AMOUNT", "SEC.DISBURSED.AMOUNT", "PRIMARY.INSTAL.AMT", "SEC.INSTAL.AMT", 
"NEW.ACCTS.IN.LAST.SIX.MONTHS", "DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS"]
model.get_booster().feature_names = x_FN
plot_importance(model.get_booster())
plt.show()

model.get_booster().feature_names = None
y_predict = model.predict(x_predict)

print(y_predict.shape) # (112392,)

df = pd.DataFrame(y_predict, index=None, columns=["loan_default"])
print(df)

dd = df.groupby("loan_default")["loan_default"].count()
print(dd)
# 0    111706
# 1       686
index = np.arange(len(dd))
label = ['0', '1']
plt.bar(index,dd)
plt.title("LOAN DEFAULT")
plt.xticks(index,label)
plt.show()

# loan_default
# 0    111706
# 1       686

model.get_booster().feature_names = x_FN
plot_importance(model.get_booster())
plt.show()