import numpy as  np
from keras.models import  Sequential
from keras.layers import  Dense

# 1. 데이터
x = np.array(range(1,101 ))
y = np.array(range(101,201))

y_pred = np.array(range(201,251) )

from sklearn.model_selection import  train_test_split

# x_train, x_test , y_train  , y_test = train_test_split(x , y ,  train_size=0.7,shuffle=false)

x_train, x_test , y_train  , y_test = train_test_split(x ,  y ,  train_size=0.6, shuffle=True)

# x_test, x_train, y_test, y_train = train_test_split(x , y ,  test_size=0.4, shuffle=False)

print(x_train )

x_val = x_train[50:70 ]
y_val = y_train[50:70 ]

# 2. 모델 구성
model = Sequential()
model.add(Dense(10,input_dim = 1) )
model.add(Dense(10) )
model.add(Dense(10) )
model.add(Dense(1) ) 


# 3. 컴파일, 훈련

model.compile(loss='mse',optimizer='adam',metrics=['mae'] )
model.fit(x_train ,y_train, epochs=1, batch_size=1,validation_split=0.2 )



# 4. 평가, 예측
loss = model.evaluate(x_test,y_test )

print("loss :",loss )

y_predict = model.predict(y_pred )
print("y_predict : ",y_predict )

x_test_predict = model.predict(x_test )

# # 회귀모델에서 검증에 사용되는 평가지표 : RMSE R2
# # RMSE
from sklearn.metrics import  mean_squared_error
def RMSE(y_test,x_test_predict ):
        return np.sqrt(mean_squared_error(y_test,x_test_predict) )
print("RMSE : ",RMSE(y_test,x_test_predict) )

# R2
from sklearn.metrics import  r2_score
r2 = r2_score(y_test,x_test_predict )
print("R2 : ",r2 )



