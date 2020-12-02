from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt

x_FN = ["disbursed_amount","asset_cost", "ltv", "Date.of.Birth","Employment.Type", "MobileNo_Avl_Flag",
"Aadhar_flag", "PAN_flag", "VoterID_flag", "Passport_flag", "PERFORM_CNS.SCORE", "PRI.NO.OF.ACCTS",
"PRI.ACTIVE.ACCTS", "PRI.OVERDUE.ACCTS", "PRI.CURRENT.BALANCE", "PRI.SANCTIONED.AMOUNT", 
"PRI.DISBURSED.AMOUNT", "SEC.NO.OF.ACCTS", "SEC.ACTIVE.ACCTS", "SEC.OVERDUE.ACCTS", "SEC.CURRENT.BALANCE",
"SEC.SANCTIONED.AMOUNT", "SEC.DISBURSED.AMOUNT", "PRIMARY.INSTAL.AMT", "SEC.INSTAL.AMT", 
"NEW.ACCTS.IN.LAST.SIX.MONTHS", "DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS"]

x = np.load("./data/loan_x.npy")
y = np.load("./data/loan_y.npy")

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=30,shuffle=True)

model = XGBClassifier()
model.fit(x_train,y_train)

acc = model.score(x_test,y_test)
print("acc : ",acc)
# acc :  0.7822264159035834


model.get_booster().feature_names = x_FN
plot_importance(model.get_booster())
plt.show()


# [0.         0.01664576 0.01690725 0.01818245 0.01932519 0.01999933
#  0.02019627 0.02419971 0.02594982 0.02645867 0.02725448 0.02906843
#  0.02936665 0.03006568 0.03431729 0.03452988 0.03964236 0.0458775
#  0.04646527 0.05172378 0.05412839 0.05449096 0.05472836 0.05858004
#  0.07074173 0.07188845 0.07926639]

thresholds = np.sort(model.feature_importances_)
print(thresholds)
n     = 0
b_acc = 0

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    

    select_x_train = selection.transform(x_train)
    selection_model = XGBClassifier()
    selection_model.fit(select_x_train,y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    acc = selection_model.score(select_x_test,y_test)
    acc_score = accuracy_score(y_test,y_predict)
    if acc > b_acc:
        n = select_x_train.shape[1]
        b_acc = acc
        L_selection = selection
        print("Thresh=%.3f, n=%d, acc: %.15f%%, acc_score: %.15f%%"%(thresh,select_x_train.shape[1],acc,acc_score))
    
x_train = L_selection.transform(x_train)
x_test = L_selection.transform(x_test)

model = XGBClassifier()
model.fit(x_train,y_train)

acc = model.score(x_test,y_test)
print("acc : ",acc)

model.get_booster().feature_names = x_FN
plot_importance(model.get_booster())
plt.show()

thresholds = np.sort(model.feature_importances_)
print(thresholds)

# 1
# acc :  0.7822264159035834
# [0.         0.01664576 0.01690725 0.01818245 0.01932519 0.01999933
#  0.02019627 0.02419971 0.02594982 0.02645867 0.02725448 0.02906843
#  0.02936665 0.03006568 0.03431729 0.03452988 0.03964236 0.0458775
#  0.04646527 0.05172378 0.05412839 0.05449096 0.05472836 0.05858004
#  0.07074173 0.07188845 0.07926639]



# 2
# acc :  0.7830198794793163
# [0.07782567 0.0813188  0.0834452  0.08665784 0.08984786 0.09546391
#  0.09887273 0.11032974 0.12164117 0.15459709]





























