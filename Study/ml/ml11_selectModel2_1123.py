# 클래스파이어 모델들을 추출

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')
boston = pd.read_csv("./data/csv/boston_house_prices.csv",header=0,index_col=0)
print(boston)

x = boston.iloc[:,1:13]
y = boston.iloc[:,0]
print("x : \n",x)
print("y : \n",y)
print("x.shape : \n",x.shape)
print("y.shape : \n",y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, random_state=55,shuffle=True)

allAlgorithms = all_estimators(type_filter='regressor') # 리그레서 모델들을 추출

for (name, algorithm) in allAlgorithms:
    model = algorithm()

    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print(name, "의 정답률 : ",r2_score(y_test,y_pred))

import sklearn
print(sklearn.__version__)