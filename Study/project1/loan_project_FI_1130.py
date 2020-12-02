from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from xgboost import XGBClassifier

x = np.load("./data/loan_x.npy")
y = np.load("./data/loan_y.npy")
x_FN = ["disbursed_amount","asset_cost", "ltv", "Date.of.Birth","Employment.Type", "MobileNo_Avl_Flag",
"Aadhar_flag", "PAN_flag", "VoterID_flag", "Passport_flag", "PERFORM_CNS.SCORE", "PRI.NO.OF.ACCTS",
"PRI.ACTIVE.ACCTS", "PRI.OVERDUE.ACCTS", "PRI.CURRENT.BALANCE", "PRI.SANCTIONED.AMOUNT", 
"PRI.DISBURSED.AMOUNT", "SEC.NO.OF.ACCTS", "SEC.ACTIVE.ACCTS", "SEC.OVERDUE.ACCTS", "SEC.CURRENT.BALANCE",
"SEC.SANCTIONED.AMOUNT", "SEC.DISBURSED.AMOUNT", "PRIMARY.INSTAL.AMT", "SEC.INSTAL.AMT", 
"NEW.ACCTS.IN.LAST.SIX.MONTHS", "DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS"]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66,shuffle=True)

model = XGBClassifier()

model.fit(x_train,y_train)

acc = model.score(x_test,y_test)

print("acc : ",acc)


print(model.feature_importances_)



# 시각화
import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importance_cancer(model):
    n_features = x.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align="center")
    plt.yticks(np.arange(n_features),x_FN)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importance_cancer(model)
plt.show()













