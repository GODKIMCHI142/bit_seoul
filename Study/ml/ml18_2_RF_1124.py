from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, train_size=0.7, random_state=66,shuffle=True)

model = RandomForestClassifier(max_depth=4)

model.fit(x_train,y_train)

acc = model.score(x_test,y_test)

print("acc : ",acc)
# acc :  0.9590643274853801

print(model.feature_importances_)
# [0.02907101 0.01444303 0.04770293 0.04335729 0.00481803 0.00308946
#  0.03466559 0.06726268 0.00166025 0.00225544 0.01941682 0.0030846
#  0.01480478 0.04792892 0.00327718 0.00384497 0.00788074 0.00214916
#  0.00430557 0.00432171 0.17881475 0.01833824 0.15029894 0.13101673
#  0.01093117 0.01496791 0.01790502 0.10426367 0.00838678 0.00573662]


# 시각화
import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importance_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align="center")
    plt.yticks(np.arange(n_features),cancer.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importance_cancer(model)
plt.show()













