from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np
from sklearn.decomposition import PCA

cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, train_size=0.7, random_state=66,shuffle=True)

print(x_train.shape)

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



ii = []
for i in range(cancer.data.shape[1]):
    if model.feature_importances_[i] == 0:
        print(i,"번째 컬럼은 0")
        ii.append(i)
    else:
        print(i," 번째 컬럼은 0아님")

print(ii) # [0, 2, 5, 25]

for i in range(len(ii)):
    x_train = np.delete(x_train,ii[i],axis=1)
    x_test  = np.delete(x_test,ii[i],axis=1)
    print(x_train.shape)
    print(x_test.shape)


# cumsum

pca = PCA()
pca.fit(x_train)
# 중요도가 높은순서대로 바뀐다.
pca.explained_variance_ratio_
xd = len(pca.explained_variance_ratio_)
print(xd)
xd = int((xd*0.7))
print(xd)
pca = PCA(n_components=xd)

x_train = pca.fit_transform(x_train)
x_test = pca.fit_transform(x_test)
model = XGBClassifier(max_depth=4)

model.fit(x_train,y_train)

acc = model.score(x_test,y_test)

print("acc : ",acc)
# acc :  0.935672514619883

print(model.feature_importances_)
# [0.32955214 0.02794053 0.16277935 0.05892859 0.04615274 0.06440601
#  0.06184852 0.06679642 0.00328083 0.02632914 0.0264015  0.01425229
#  0.02749873 0.02518361 0.02742266 0.0093743  0.01368772 0.00816491]










