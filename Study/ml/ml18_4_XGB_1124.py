from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, train_size=0.7, random_state=66,shuffle=True)

model = XGBClassifier(max_depth=4)

model.fit(x_train,y_train)

acc = model.score(x_test,y_test)

print("acc : ",acc)
# acc :  0.9766081871345029

print(model.feature_importances_)
# [0.         0.0298866  0.         0.0331422  0.00364928 0.
#  0.00768587 0.0259198  0.00152558 0.00107367 0.01166475 0.00157133
#  0.01476577 0.00808299 0.00318396 0.00500426 0.00037138 0.00296827
#  0.00284988 0.00421732 0.25828916 0.01033985 0.30329755 0.19179387
#  0.00225331 0.         0.01109839 0.056079   0.00396034 0.00532566]

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














