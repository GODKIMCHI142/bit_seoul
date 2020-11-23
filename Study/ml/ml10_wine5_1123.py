# winequality-white.csv 
# RandomForest로 모델 만들기
# 분류모델인듯

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score




# 1. data
dataset = pd.read_csv("./data/csv/winequality-white.csv",header=0,index_col=None,sep=";")
print(dataset.describe())

count_data = dataset.groupby("quality")["quality"].count()
# 퀄리티 컬럼에 있는 개체의 이름을 quality로 하고 groupby하여 카운트 하겠다
print(count_data)
# quality
# 3      20
# 4     163
# 5    1457
# 6    2198
# 7     880
# 8     175
# 9       5

datasets = pd.DataFrame(dataset)
x = datasets.iloc[:,:11]
y = datasets.iloc[:,11]


newlist = []
for i in list(y):
    if   i == 8:
        newlist += [7]
    if   i == 3:
        newlist += [3]
    if   i == 4:
        newlist += [4]
    if   i == 5:
        newlist += [5]
    if   i == 6:
        newlist += [6]
    if   i == 7:
        newlist += [7] 
    if   i == 9:
        newlist += [9]

y = newlist
x = np.array(x)
y = np.array(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, random_state=55,shuffle=True)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(y_test)


scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)
print(y_train.shape)
print(y_test.shape)

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64,input_shape=(11,),activation='relu')) 
model.add(Dense(32,activation='relu'))  
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(3,activation='softmax'))                       
model.summary()               

print(x_train.shape)
print(x_test.shape)



# 3. 훈련
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=100, mode='auto')

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics="acc")
model.fit(x_train,y_train,epochs=10000,batch_size=16,validation_split=0.3,callbacks=[es],verbose=1)


# 4. 평가 예측
loss , acc= model.evaluate(x_test,y_test,batch_size=16)

print("loss : ",loss)
print("acc : ",acc)

x_pred = x_test[:10]
y_real = y_test[:10]

# Y_class_recovery = np.argmax(Y_class_onehot, axis=1).reshape(-1,1)
y_test_predict = model.predict([x_pred])

ytp_recovery = np.argmax(y_test_predict,axis=1)
print("예측값 :",ytp_recovery)

y_real = np.argmax(y_real,axis=1)
print("실제값 :",y_real)

# model_score = model.score(x_test,y_test)
# print("model_score : ",model_score)

acc_score = accuracy_score(y_real,ytp_recovery)
print("acc_score :   ",acc_score)


# loss :  0.2940368950366974
# acc :  0.9948979616165161
# 예측값 : [1 1 1 1 1 1 1 1 1 1]
# 실제값 : [1 1 1 1 1 1 1 1 1 1]
# acc_score :    1.0








