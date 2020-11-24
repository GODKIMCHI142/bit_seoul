from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, train_size=0.7, random_state=66,shuffle=True)

model = GradientBoostingClassifier(max_depth=4)

model.fit(x_train,y_train)

acc = model.score(x_test,y_test)

print("acc : ",acc)
# acc :  0.9532163742690059

print(model.feature_importances_)
# [1.40007696e-04 6.16173165e-02 5.26415859e-05 1.60147457e-03
#  2.15418858e-03 4.33379371e-05 7.35806493e-04 1.00663494e-02
#  9.33367137e-05 7.78383564e-03 8.55258207e-03 1.57550199e-03
#  2.04130135e-04 1.76190263e-02 1.29569887e-03 6.12881461e-05
#  1.51097871e-03 4.38085436e-03 1.81168178e-03 3.12928078e-03
#  3.76958763e-01 1.99104063e-02 3.68725743e-04 3.70235672e-01
#  3.82233933e-03 4.30589602e-05 4.69561866e-03 9.86085809e-02
#  2.68473014e-04 6.59043559e-04]

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
















