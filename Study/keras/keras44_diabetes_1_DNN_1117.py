# 회귀모델
import numpy as np

# 1. 데이터 준비
from sklearn.datasets import load_diabetes
dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x.shape) # (442,10)
print(y.shape) # (442,)

# 데이터 전처리
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler 
# from sklearn.preprocessing import RobustScaler
# scaler = StandardScaler()
# scaler.fit(x)

# fit한 결과로 transform
x = scaler.transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test , y_train  , y_test = train_test_split(x , y , train_size=0.8, random_state=1)


# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

model = Sequential()
model.add(Dense(30,input_shape=(10,),activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(70,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

model.summary()

print(x.shape)
print(y.shape)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
# 3. 컴파일 훈련

# ES
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=100, mode='auto')

model.compile(loss="mse",optimizer="adam",metrics="mae")
model.fit(x_train,y_train,epochs=10000,batch_size=10,validation_split=0.2,callbacks=[es],verbose=1)

print("fit end")
# 4. 평가 예측
loss = model.evaluate(x_test,y_test,batch_size=10)

print("loss : ",loss)

# predict
x_pred = x_test[0:10]
y_pred = y_test[0:10]

y_test_predict = model.predict([x_pred])

print("예측값 :",y_test_predict)
print("실제값 :",y_pred)

# RMSE 
from sklearn.metrics import mean_squared_error
def RMSE(y_test_R,y_test_predict_R):
        y_t     = y_test_R
        y_t_pre = y_test_predict_R
        return np.sqrt(mean_squared_error(y_t,y_t_pre))
print("RMSE : ",RMSE(y_pred,y_test_predict)) 

# R2
from sklearn.metrics import r2_score
# r2 = r2_score(y_test,x_test_predict)
print("R2 : ",r2_score(y_pred,y_test_predict))


# 9/9 [==============================] - 0s 1ms/step - loss: 5059.1084 - mae: 53.8015
# loss :  [5059.1083984375, 53.80153274536133]
# 예측값 : [[103.96419 ]
#  [ 62.100216]
#  [128.60428 ]
#  [ 73.09948 ]
#  [217.08485 ]
#  [169.21907 ]
#  [272.51465 ]
#  [ 82.002396]
#  [155.96213 ]
#  [122.8061  ]]
# 실제값 : [ 78. 152. 200.  59. 311. 178. 332. 132. 156. 135.]
# RMSE :  53.97811971868136
# R2 :  0.5893630659307498


