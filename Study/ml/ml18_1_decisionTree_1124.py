from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, train_size=0.7, random_state=66,shuffle=True)

model =  DecisionTreeClassifier(max_depth=4)

model.fit(x_train,y_train)

acc = model.score(x_test,y_test)

print("acc : ",acc)
# acc :  0.9415204678362573
print(model.feature_importances_)
# [0.         0.02869086 0.         0.         0.         0.
#  0.         0.         0.         0.03043354 0.         0.
#  0.         0.01481385 0.01502854 0.         0.         0.
#  0.         0.         0.         0.01859782 0.         0.77570942
#  0.         0.         0.         0.11672597 0.         0.        ]

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















