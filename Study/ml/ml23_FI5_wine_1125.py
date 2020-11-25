# 기준 XGB
# 1. feature_importance 0 인놈 제거 % train기준
# 2. 하위 30% 제거
# 3. 디폴트와 성능비교



from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np
from sklearn.decomposition import PCA

iris = load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, train_size=0.7, random_state=66,shuffle=True)

print(x_train.shape)

model = XGBClassifier(max_depth=4)

model.fit(x_train,y_train)

acc = model.score(x_test,y_test)

print("acc : ",acc)
# acc :  0.9111111111111111

print(model.feature_importances_)


ii = []
for i in range(iris.data.shape[1]):
    if model.feature_importances_[i] == 0:
        print(i,"번째 컬럼은 0")
        ii.append(i)
    else:
        print(i," 번째 컬럼은 0아님")

print(ii) # [0, 2, 5, 25]

if len(ii) >= 1:
    print("0 제거 for문")
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


print(model.feature_importances_)
# acc :  0.8666666666666667










