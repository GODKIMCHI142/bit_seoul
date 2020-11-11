
import numpy as np

# 1. 데이터
# x1 = np.array([range(1,101),range(311,411),range(100)])
# y1 = np.array([range(101,201),range(711,811),range(100)])
# print("x1.shape : ",x1.shape) # (3,100)

x = np.array([range(1,101),range(311,411),range(100)])
y = np.array(range(101,201))
print("x.shape : ",x.shape)  # (3,100)

x = np.transpose(x) 
# y = np.transpose(y) 
print("x.shape : ",x.shape) # (100,3)
print("y.shape : ",y.shape) # (100,3)

# splice
# x_train = x[:len1]
# y_train = y[:len1]
# x_val = x[len1:len2]
# y_val = y[len1:len2]
# x_test = x[len2:]
# y_test = y[len2:]

# print(x_train.shape) 
# print(y_train.shape)
# print(x_val.shape) 
# print(y_val.shape)
# print(x_test.shape)
# print(y_test.shape)
# y1, y2, y3 = w1x1 + w2x2 + w3x3 + b

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input

input1 = Input(shape=(3, ))
dense1 = Dense(5, activation='relu')(input1) # 위에 있는 input레이어를 사용하겠다.
dense2 = Dense(4, activation='relu')(dense1)
dense3 = Dense(3, activation='relu')(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs=input1,outputs=output1)

# activation(활성화함수) : 가중치 계산할 때 너무 큰값이나 작은값을 걸러준다.
#                       : 가중치를 조정해주기 위해서 활성화함수를 사용한다.
#                       : 모든 레이어마다 디폴트가 있다. 
#                       : relu를 자주사용한다.
#                       : Dense층에서는 'linear'가 default값이다.

# model = Sequential()
# model.add(Dense(5,input_shape=(3, ), activation='relu'))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(3, activation='relu'))
# model.add(Dense(1))


model.summary()










'''
# 3. 컴파일 훈련

from sklearn.model_selection import train_test_split
x_train, x_test , y_train  , y_test = train_test_split(x , y , train_size=0.6, shuffle=True)
x_val, x_test , y_val  , y_test = train_test_split(x , y , test_size=0.5, shuffle=True)


model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train ,y_train, epochs=100, batch_size=1,validation_data=(x_val,y_val), verbose=2)


# 4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print("loss :",loss)

x_test_predict = model.predict(x_test)
print("y_test :",y_test)
print("x_test_predict :",x_test_predict)

# RMSE 
from sklearn.metrics import mean_squared_error
def RMSE(y_test,x_test_predict):
        return np.sqrt(mean_squared_error(y_test,x_test_predict))
print("RMSE : ",RMSE(y_test,x_test_predict)) 


# R2
from sklearn.metrics import r2_score
r2 = r2_score(y_test,x_test_predict)
print("R2 : ",r2)



# x_len = x.__len__()
# print("x.__len__() : ",x.__len__())

# new_x = []
# np.array(new_x)
# for i in range(x[1,].size):
#     # print("s",x[1,].size)
#     # new_x += [x[0,i],x[1,i],x[2,i]]
#     # on = np.array(x[0,i])
#     # tw = np.array(x[1,i])
#     # th = np.array(x[2,i])
#     # new_x = np.append(new_x,[on],axis=0)
#     # new_x = np.append(new_x,[tw],axis=0)
#     # new_x = np.append(new_x,[th],axis=0)
#     new_x.append([x[0,i],x[1,i],x[2,i]])
# new_x = np.array(new_x)
# print(new_x)
# print("new_x.shape >>> ",new_x.shape)

'''






