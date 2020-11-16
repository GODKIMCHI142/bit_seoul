from numpy import array
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
# 1. 데이터 
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],[2000,3000,4000],[3000,4000,5000],[4000,5000,6000],[100,200,300]]) 
           #(14,3) 
y = array([4,5,6,7,8,9,10,11,12,13,5000,6000,7000,400])

x_predict = array([55,65,75])
x_predict2 = array([6600,6700,6800]) # => transform에 넣는다.

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(x)
x = scaler.transform(x)
print("x : >>>>> ",x)
# scaler.fit(x_predict)
x_predict = x_predict.reshape(1,3)
x_predict2 = x_predict2.reshape(1,3)
print("x.shape : >>>>> ",x_predict.shape)
x_predict  = scaler.transform(x_predict)
x_predict2 = scaler.transform(x_predict2)

print(x.shape)
x = x.reshape(14,3,1)
y = y.reshape(14,1)


from sklearn.model_selection import train_test_split
x_train, x_test , y_train  , y_test = train_test_split(x , y , train_size=0.8)
x_val, x_test , y_val  , y_test = train_test_split(x_test , y_test , test_size=0.5,random_state=1)


# 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(90,activation='relu',input_shape=(3,1)))
model.add(Dense(80,activation='relu'))
model.add(Dense(70,activation='relu'))
model.add(Dense(60,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

model.summary()

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor='loss',patience=10, mode='min')
early_stopping = EarlyStopping(monitor='loss',patience=100, mode='auto')

model.compile(loss='mse',optimizer='adam',metrics='mae')
history = model.fit(x_train,y_train,epochs=1000,batch_size=1,
                    callbacks=[early_stopping],validation_data=(x_val,y_val))

# print("model.fit.history : >>>>> \n",history) # tensorflow.python.keras.callbacks.History
# print("history.history.keys() : >>>> \n",history.history.keys())
# print("history.history[loss] : >>>> \n",history.history['loss'])
# print("history.history[val_loss] : >>>> \n",history.history['val_loss'])

# 그래프
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])

plt.title('loss & mae')
plt.ylabel('loss , mae')
plt.xlabel('epochs')

plt.legend(['train loss','val loss','train mae','val mae'])
plt.show()

# 4. 평가, 예측

loss , mse = model.evaluate(x_test,y_test)
print("loss : ",loss)
print("mse : ",mse)

# x_pred = np.array([97,98,99,100])
x_predict = x_predict.reshape(1,3,1)
x_pred = model.predict(x_predict)
print("x_pred : ",x_pred)

# loss :  1.2178049087524414
# mse :  0.9814658164978027
# x_pred :  [[8.512883]]







