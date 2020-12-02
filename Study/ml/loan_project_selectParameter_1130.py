learning_rate  = [0.05, 0.06, 0.07, 0.08, 0.09]
n_estimators = [200, 300, 400, 500]
max_depth = [6,8,10]
colsample_bytree = [0.6, 0.7, 0.8]
colsample_bylevel = [0.6, 0.7, 0.8]


from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

x         = np.load("./data/loan_x.npy") 
x_predict = np.load("./data/loan_x_predict.npy") # (112392, )
y         = np.load("./data/loan_y.npy")

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=30,shuffle=True)

parameters2 = [
    {
    "svm__learning_rate" : [0.05, 0.06, 0.07, 0.08, 0.09],
    "svm__n_estimators" : [200, 300, 400, 500],
    "svm__max_depth" : [6,8,10],
    "svm__colsample_bytree" : [0.6, 0.7, 0.8],
    "svm__colsample_bylevel" : [0.6, 0.7, 0.8],
    }
]



parameters = dict(learning_rate=learning_rate, n_estimators=n_estimators, 
max_depth = [6,8,10,12,15], colsample_bylevel = [0.6, 0.7, 0.8], colsample_bytree = [0.6, 0.7, 0.8])

# parameters2 = dict(learning_rate=learning_rate2, n_estimators=n_estimators2, 
# max_depth =max_depth2, colsample_bylevel = colsample_bytree2, colsample_bytree = colsample_bylevel2)


model = RandomizedSearchCV(XGBClassifier(),parameters,cv=3,verbose=2)
model.fit(x_train,y_train)
acc = model.score(x_test,y_test)
print("acc : ",acc)

model = model.best_estimator_

# x_FN = ["disbursed_amount","asset_cost", "ltv", "Date.of.Birth","Employment.Type", "MobileNo_Avl_Flag",
# "Aadhar_flag", "PAN_flag", "VoterID_flag", "Passport_flag", "PERFORM_CNS.SCORE", "PRI.NO.OF.ACCTS",
# "PRI.ACTIVE.ACCTS", "PRI.OVERDUE.ACCTS", "PRI.CURRENT.BALANCE", "PRI.SANCTIONED.AMOUNT", 
# "PRI.DISBURSED.AMOUNT", "SEC.NO.OF.ACCTS", "SEC.ACTIVE.ACCTS", "SEC.OVERDUE.ACCTS", "SEC.CURRENT.BALANCE",
# "SEC.SANCTIONED.AMOUNT", "SEC.DISBURSED.AMOUNT", "PRIMARY.INSTAL.AMT", "SEC.INSTAL.AMT", 
# "NEW.ACCTS.IN.LAST.SIX.MONTHS", "DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS"]
# model.get_booster().feature_names = x_FN
# plot_importance(model.get_booster())
# plt.show()



# print(thresholds)


n     = 0
b_acc = 0

thresholds = np.sort(model.feature_importances_)
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
x_predict = L_selection.transform(x_predict)

pipe = Pipeline([("scaler",MinMaxScaler()), ("svm",XGBClassifier())])
model = RandomizedSearchCV(pipe,parameters2,cv=3,verbose=2)
model.fit(x_train,y_train)

acc = model.score(x_test,y_test)
print("acc : ",acc) # acc :  0.7829340996332912

model = model.best_estimator_

# model.get_booster().feature_names = x_FN
# plot_importance(model.get_booster())
# plt.show()

# thresholds = np.sort(model.feature_importances_)
# print(thresholds)

# model.get_booster().feature_names = None


y_predict = model.predict(x_test)

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




# acc :  0.7832557740558855
# acc :  0.7831485492483541
# [0.         0.02073999 0.02111412 0.02296875 0.02418768 0.02535132
#  0.02561702 0.02623364 0.02880394 0.0295934  0.03136835 0.03243126
#  0.03397909 0.03537619 0.03567686 0.03686796 0.03808521 0.04268289
#  0.0427559  0.04420536 0.05009787 0.0523834  0.05367599 0.05910351
#  0.06049902 0.06142915 0.06477211]



# acc :  0.7830627694023289
# [0.0708025  0.08016055 0.08828182 0.08964958 0.10360795 0.10418013
#  0.10885821 0.11216553 0.12089268 0.12140108]




# acc :  0.78299843451781
# [0.         0.01568703 0.01571819 0.01642605 0.01873519 0.01940226
#  0.02121342 0.02304367 0.02464764 0.02521022 0.02928036 0.02973921
#  0.02976303 0.03177422 0.03239527 0.03823703 0.03832289 0.04042343
#  0.0432143  0.05416643 0.05543635 0.05728289 0.06042759 0.06366269
#  0.06986641 0.06996395 0.07596029]



















