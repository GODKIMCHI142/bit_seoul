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

print(model.feature_importances_)
# [0.         0.0298866  0.         0.0331422  0.00364928 0.
#  0.00768587 0.0259198  0.00152558 0.00107367 0.01166475 0.00157133
#  0.01476577 0.00808299 0.00318396 0.00500426 0.00037138 0.00296827
#  0.00284988 0.00421732 0.25828916 0.01033985 0.30329755 0.19179387
#  0.00225331 0.         0.01109839 0.056079   0.00396034 0.00532566]

# [0.         0.01560227 0.         0.01683669 0.00414788 0.00349809
#  0.01892252 0.05857253 0.00052208 0.00340979 0.00686431 0.
#  0.00935022 0.00415534 0.00284723 0.00331549 0.01119763 0.00749157
#  0.00111885 0.00082988 0.5000727  0.01805239 0.19511884 0.02013689
#  0.00487486 0.00333159 0.01238519 0.07613306 0.00121204 0.        ]


# ii = []
# for i in range(cancer.data.shape[1]):
#     if model.feature_importances_[i] == 0:
#         print(i,"번째 컬럼은 0")
#         ii.append([i])
#     else:
#         print(i," 번째 컬럼은 0아님")

# print(ii) [[0], [2], [5], [25]]
# cumsum

x = np.append(x_train, x_test, axis=0)
x = x.reshape(x.shape[0],(x.shape[1]*x.shape[2])) # 7만, 784
y = np.append(y_train, y_test, axis=0)

pca = PCA()
pca.fit(x)
# 중요도가 높은순서대로 바뀐다.
cumsum = np.cumsum(pca.explained_variance_ratio_)
# cumsum : 누적된 합
d = np.argmax(cumsum >= 0.95) + 1
pca = PCA(n_components=d)

x = pca.fit_transform(x)
print(x.shape)# 70000, 154
print(y.shape)# 70000, 154

from sklearn.model_selection import train_test_split
x_train, x_test , y_train  , y_test = train_test_split(x , y , train_size=0.8, random_state=1)



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














