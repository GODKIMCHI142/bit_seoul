
import numpy as np

# 1. 데이터
# x1 = np.array([range(1,101),range(311,411),range(100)])
# y1 = np.array([range(101,201),range(711,811),range(100)])
# print("x1.shape : ",x1.shape) # (3,100)

x1 = np.array([range(1,101),range(311,411),range(100)])
y1 = np.array([range(101,201),range(711,811),range(100)])
print("x1.shape : ",x1.shape)  # (3,100)
print("y1.shape : ",y1.shape)  # (3,100)

x1 = np.transpose(x1) 
y1 = np.transpose(y1) 

print("x1.shape : ",x1.shape) # (100,3)
print("y1.shape : ",y1.shape) # (100,3)

x2 = np.array([range(1,101),range(311,411),range(100)])
y2 = np.array([range(501,601),range(431,531),range(100,200)])
print("x2.shape : ",x2.shape)  # (3,100)
print("y2.shape : ",y2.shape)  # (3,100)

x2 = np.transpose(x2) 
y2 = np.transpose(y2) 
print("x2.shape : ",x2.shape) # (100,3)
print("y2.shape : ",y2.shape) # (100,3)

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


# 데이터 나누기
from sklearn.model_selection import train_test_split
x1_train, x1_test , y1_train  , y1_test = train_test_split(x1 , y1 , train_size=0.6, shuffle=True)
x1_val, x1_test , y1_val  , y1_test = train_test_split(x1 , y1 , test_size=0.5, shuffle=True)

x2_train, x2_test , y2_train  , y2_test = train_test_split(x2 , y2 , train_size=0.6, shuffle=True)
x2_val, x2_test , y2_val  , y2_test = train_test_split(x2 , y2 , test_size=0.5, shuffle=True)


# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

# Model 1
input1 = Input(shape=(3, ))
dense1_1 = Dense(10, activation='relu',name='dense1_1')(input1) 
dense2_1 = Dense(9, activation='relu',name='dense1_2')(dense1_1)
dense3_1 = Dense(8, activation='relu',name='dense1_3')(dense2_1)
output1 = Dense(3,name='output1')(dense3_1)

# Model 2
input2 = Input(shape=(3, ))
dense2_1 = Dense(7, activation='relu',name='dense2_1')(input2) 
dense2_2 = Dense(6, activation='relu',name='dense2_2')(dense2_1)
output2 = Dense(3,name='output2')(dense2_2)

# 모델 병합, concatenate
from tensorflow.keras.layers import concatenate,Concatenate
# from keras.layers.merge import concatenate,Concatenate
# from keras.layers import concatenate,Concatenate

# merge1 = concatenate([output1, output2])
merge1 = Concatenate(axis=0)([output1,output2])

middle1_1 = Dense(30,name='middle1_1')(merge1)
middle1_2 = Dense(10,name='middle1_2')(middle1_1)
middle1_3 = Dense(20,name='middle1_3')(middle1_2)

# output 모델 구성(분기)
output1_1 = Dense(30,name='output1_1')(middle1_3)
output1_2 = Dense(20,name='output1_2')(output1_1)
output1_3 = Dense(3,name='output1_3')(output1_2)


output2_1 = Dense(30,name='output2_1')(middle1_3)
output2_2 = Dense(20,name='output2_2')(output2_1)
output2_3 = Dense(10,name='output2_3')(output2_2)
output2_4 = Dense(3,name='output2_4')(output2_3)

# 모델 정의
# model1 = Model(inputs=input1,outputs=output1,name="Model1")
# model2 = Model(inputs=inputt1,outputs=outputt1,name="Model2")
# model1.summary()
# model2.summary()

model = Model(inputs=[input1,input2],outputs=[output1_3,output2_4])

model.summary()




# 3. 컴파일 훈련

# from sklearn.model_selection import train_test_split
# x_train, x_test , y_train  , y_test = train_test_split(x , y , train_size=0.6, shuffle=True)
# x_val, x_test , y_val  , y_test = train_test_split(x , y , test_size=0.5, shuffle=True)


model.compile(loss='mse', optimizer='adam', metrics='mse')
model.fit([x1_train ,x2_train],[y1_train ,y2_train], epochs=1, 
           batch_size=1,validation_data=([x1_val,x2_val],[y1_val,y2_val]))


# 4. 평가, 예측
result = model.evaluate(([x1_test ,x2_test]),([y1_test ,y2_test]),batch_size=1)
print("result :",result)

# x_test_predict = model.predict(x_test)
# print("y_test :",y_test)
# print("x_test_predict :",x_test_predict)

# # RMSE 
# from sklearn.metrics import mean_squared_error
# def RMSE(y_test,x_test_predict):
#         return np.sqrt(mean_squared_error(y_test,x_test_predict))
# print("RMSE : ",RMSE(y_test,x_test_predict)) 


# # R2
# from sklearn.metrics import r2_score
# r2 = r2_score(y_test,x_test_predict)
# print("R2 : ",r2)



# # x_len = x.__len__()
# # print("x.__len__() : ",x.__len__())

# # new_x = []
# # np.array(new_x)
# # for i in range(x[1,].size):
# #     # print("s",x[1,].size)
# #     # new_x += [x[0,i],x[1,i],x[2,i]]
# #     # on = np.array(x[0,i])
# #     # tw = np.array(x[1,i])
# #     # th = np.array(x[2,i])
# #     # new_x = np.append(new_x,[on],axis=0)
# #     # new_x = np.append(new_x,[tw],axis=0)
# #     # new_x = np.append(new_x,[th],axis=0)
# #     new_x.append([x[0,i],x[1,i],x[2,i]])
# # new_x = np.array(new_x)
# # print(new_x)
# # print("new_x.shape >>> ",new_x.shape)








