# 클래스파이어 모델들을 추출

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')
iris = pd.read_csv("./data/csv/iris_ys.csv",header=0,index_col=0)

x = iris.iloc[:,0:4]
y = iris.iloc[:,4]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, random_state=55,shuffle=True)

allAlgorithms = all_estimators(type_filter='classifier') # 클래스파이어 모델들을 추출

for (name, algorithm) in allAlgorithms:
    model = algorithm()

    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print(name, "의 정답률 : ",accuracy_score(y_test,y_pred))

import sklearn
print(sklearn.__version__)