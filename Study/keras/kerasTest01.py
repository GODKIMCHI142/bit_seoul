from keras.layers import Dense
from keras.models import Sequential
import numpy as np


# input 2 output 1 함수모델

# 1. 데이터 준비
x1 = np.array([range(100),range(101,201),range(301,401)])
x2 = np.array([range(100),range(101,201),range(301,401)])
y1 = np.array([range(100),range(100),range(100)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
print("x1 : x2 : y1\n",x1.shape,x2.shape,y1.shape)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1,y1,train_size=0.6,shuffle=True,random_state=1)
x1_val, x1_test, y1_val, y1_test = train_test_split(x1_train,y1_train , test_size=0.5,shuffle=True,random_state=1)

x2_train, x2_test, y1_train, y1_test = train_test_split(x2,y1,train_size=0.6,shuffle=True,random_state=1)
x2_val, x2_test, y1_val, y1_test = train_test_split(x2_train,y1_train , test_size=0.5,shuffle=True,random_state=1)

print("data")

# 2. 모델 구성
from keras.models import Model 
from keras.layers import Input, concatenate


# Model 1
input1 = Input(shape=(3,))
dense1_1 = Dense(10,activation='relu',name='dense1_1')(input1)
dense1_2 = Dense(10,activation='relu',name='dense1_2')(dense1_1)
dense1_3 = Dense(10,activation='relu',name='dense1_3')(dense1_2)
output1  = Dense(10,activation='relu',name="output1")(dense1_3)

# Model 2
input2 = Input(shape=(3,))
dense2_1 = Dense(10,activation='relu',name='dense2_1')(input2)
dense2_2 = Dense(10,activation='relu',name='dense2_2')(dense2_1)
dense2_3 = Dense(10,activation='relu',name='dense2_3')(dense2_2)
output2  = Dense(10,activation='relu',name="output2")(dense2_3)

# merge
merge1 = concatenate([output1,output2])

# Middle 1
middle1_1 = Dense(10,activation='relu',name='middle1_1')(merge1)
middle1_2 = Dense(10,activation='relu',name='middle1_2')(middle1_1)
middle1_3 = Dense(10,activation='relu',name='middle1_3')(middle1_2)

# output
output1_1 = Dense(10,activation='relu',name='output1_1')(middle1_3)
output1_2 = Dense(3,activation='linear',name='output1_2')(output1_1)

# 모델 정의
model = Model(inputs=([input1,input2]),outputs=output1_2)
model.summary()

print("2")

# 3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam',metrics='mse')
model.fit(([x1_train,x2_train]),([y1_train,y1_train]),epochs=100
           ,validation_data=([x1_val,x2_val],[y1_val,y1_val]), batch_size=1,verbose=2)
print("3")

# 4. 평가, 예측
loss = model.evaluate(([x1_test,x2_test],[y1_test,y1_test]),batch_size=1)
print("loss : ",loss)
print("4")

y_test_predict = model.predict([x1_test,x2_test])
print("y_test_predict : ",y_test_predict)
#RMSE
from keras.metrics import mean_squared_error
def RMSE (y_t,y_t_pred):
    return np.sqrt(mean_squared_error(y_t,y_t_pred))
print("RMSE : ",RMSE(y1_test,y_test_predict))


#R2
from sklearn.metrics import r2_score
r2 = r2_score(y1_test,y_test_predict)
print("r2 : ",r2)




