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


datasets = pd.DataFrame(dataset)
# datasets = datasets.values
x = datasets[:,:11]
y = datasets[:,11:]



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
# 
print(y_train.shape)
print(y_test.shape)

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64,input_shape=(11,),activation='relu')) 
model.add(Dense(32,activation='relu'))  
model.add(Dense(32,activation='relu'))  
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(10,activation='softmax'))                       
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


# model.add(Dense(64,input_shape=(11,))) 
# model.add(Dense(64,activation='relu'))  
# model.add(Dense(32,activation='relu'))  
# model.add(Dense(32,activation='relu'))  
# model.add(Dense(16,activation='relu'))
# model.add(Dense(10,activation='softmax'))                       
# loss :  5.532707214355469
# acc :  0.5801020264625549
# 예측값 : [7 6 5 5 7 6 5 6 6 6]
# 실제값 : [7 6 5 5 7 6 5 6 6 6]
# acc_score :    1.0


# model.add(Dense(64,input_shape=(11,))) 
# model.add(Dense(64,activation='relu'))  
# model.add(Dense(64,activation='relu'))  
# model.add(Dense(32,activation='relu'))  
# model.add(Dense(32,activation='relu'))  
# model.add(Dense(32,activation='relu'))  
# model.add(Dense(16,activation='relu'))
# model.add(Dense(16,activation='relu'))
# model.add(Dense(10,activation='softmax'))    
# loss :  3.2815001010894775
# acc :  0.5811224579811096
# 예측값 : [7 6 5 5 7 6 6 8 7 6]
# 실제값 : [7 6 5 5 7 6 5 6 6 6]
# acc_score :    0.7








