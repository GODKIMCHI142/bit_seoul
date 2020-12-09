from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np
from sklearn.decomposition import PCA

cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, train_size=0.7, random_state=66,shuffle=True)



model = XGBClassifier(max_depth=4)

model.fit(cancer.data,cancer.target)

acc = model.score(x_test,y_test)

print("acc : ",acc)
# acc :  0.9766081871345029


import pickle
pickle.dump(model,open("./save/xgb_save/ml34_cancer.pickle.dat","wb"))
print("save")

model2 = pickle.load(open("./save/xgb_save/ml34_cancer.pickle.dat","rb"))
print("load")
acc2 = model2.score(x_test,y_test)
print("acc2 : ",acc2)




