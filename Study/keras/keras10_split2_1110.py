import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x = np.array(range(1,101))
y = np.array(range(101,201))

x_train = x[:60]
y_train = y[:60]

x_val = x[60:80]
y_val = y[60:80]

x_test = x[80:]
y_test = y[80:]

y_pred = np.array(range(201,301))


# 2. 모델 구성
model = Sequential()
model.add(Dense(10,input_dim = 1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train ,y_train, epochs=100, batch_size=1, data=(x_val,y_val))

# 4. 평가, 예측
loss = model.evaluate(x_test,y_test,batch_size=1)
print("loss :",loss)

y_predict = model.predict(y_pred)
print("y_predict : ",y_predict)



x_test_predict = model.predict(x_test)

# # 회귀모델에서 검증에 사용되는 평가지표 : RMSE R2
# # RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test,x_test_predict):
        return np.sqrt(mean_squared_error(y_test,x_test_predict))
print("RMSE : ",RMSE(y_test,x_test_predict))

# R2
from sklearn.metrics import r2_score
r2 = r2_score(y_test,x_test_predict)
print("R2 : ",r2)













