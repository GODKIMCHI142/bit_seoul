import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x = np.array(range(1,101))
y = np.array(range(101,201))
print("y : ",y)
x_train = x[:70]
y_train = y[:70]

x_test = x[70:]
y_test = y[70:]
print(x[70:])

y_pred = np.array(range(201,251))



# 2. 모델 구성
model = Sequential()
model.add(Dense(10,input_dim = 1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train ,y_train, epochs=100, batch_size=1, validation_split=0.2)


# 4. 평가, 예측
loss = model.evaluate(x_test,y_test)

print("loss :",loss)

y_predict = model.predict(y_pred)

print("y_predict : ",y_predict)

# # 회귀모델에서 검증에 사용되는 평가지표 : RMSE R2
# # RMSE
# from sklearn.metrics import mean_squared_error
# def RMSE(y_test,y_predict):
#         return np.sqrt(mean_squared_error(y_test,y_predict))
# print("RMSE : ",RMSE(y_test,y_predict))

# # R2
# from sklearn.metrics import r2_score
# r2 = r2_score(y_test,y_predict)
# print("R2 : ",r2)




