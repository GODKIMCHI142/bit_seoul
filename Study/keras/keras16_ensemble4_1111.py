# 실습 인풋2 아웃풋3


import numpy as np

# 1. 데이터
# x1 = np.array([range(1,101),range(311,411),range(100)])
# y1 = np.array([range(101,201),range(711,811),range(100)])
# print("x1.shape : ",x1.shape) # (3,100)

x1 = np.array([range(1,101),range(311,411),range(100)])

y1 = np.array([range(101,201),range(711,811),range(100)])
y2 = np.array([range(501,601),range(431,531),range(100,200)])
y3 = np.array([range(701,801),range(631,731),range(300,400)])

x1 = np.transpose(x1)
y1 = np.transpose(y1) 
y2 = np.transpose(y2)
y3 = np.transpose(y3)
print("x1.shape : ",x1.shape)
print("y1.shape : ",y1.shape)
print("y2.shape : ",y2.shape)
print("y3.shape : ",y3.shape)

# 데이터 나누기

from sklearn.model_selection import train_test_split
x1_train, x1_test , y1_train  , y1_test = train_test_split(x1 , y1 , train_size=0.6, shuffle=True)
x1_val, x1_test , y1_val  , y1_test = train_test_split(x1_test , y1_test , test_size=0.5, shuffle=True)

x1_train, x1_test , y2_train  , y2_test = train_test_split(x1 , y2 , train_size=0.6, shuffle=True)
x1_val, x1_test , y2_val  , y2_test = train_test_split(x1_test , y2_test , test_size=0.5, shuffle=True)

x1_train, x1_test , y3_train  , y3_test = train_test_split(x1 , y3 , train_size=0.6, shuffle=True)
x1_val, x1_test , y3_val  , y3_test = train_test_split(x1_test , y3_test , test_size=0.5, shuffle=True)

print("x1.shape : ",x1.shape)
print("y1.shape : ",y1.shape)
print("y2.shape : ",y2.shape)
print("y3.shape : ",y3.shape)

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

# Model 1
input1 = Input(shape=(3, ))
dense1_1 = Dense(10, activation='relu',name='dense1_1')(input1) 
dense2_1 = Dense(10, activation='relu',name='dense1_2')(dense1_1)
dense3_1 = Dense(10, activation='relu',name='dense1_3')(dense2_1)
output1 = Dense(10,name='output1')(dense3_1)

# middle
middle1_1 = Dense(10,name='middle1_1', activation='relu')(output1)
middle1_2 = Dense(10,name='middle1_2', activation='relu')(middle1_1)
middle1_3 = Dense(10,name='middle1_3', activation='relu')(middle1_2)
middle1_4 = Dense(10,name='middle1_4', activation='relu')(middle1_3)
middle1_5 = Dense(10,name='middle1_5', activation='relu')(middle1_4)
middle1_6 = Dense(10,name='middle1_6', activation='relu')(middle1_5)
middle1_7 = Dense(10,name='middle1_7', activation='relu')(middle1_6)
middle1_10 = Dense(10,name='middle1_10', activation='relu')(middle1_7)

# output 모델 구성(분기)
# 1
output1_1 = Dense(10,name='output1_1', activation='relu')(middle1_10)
output1_2 = Dense(10,name='output1_2', activation='relu')(output1_1)
output1_3 = Dense(10,name='output1_3', activation='relu')(output1_2)
output1_4 = Dense(10,name='output1_4', activation='relu')(output1_3)
output1_5 = Dense(10,name='output1_5', activation='relu')(output1_4)
output1_6 = Dense(10,name='output1_6', activation='relu')(output1_5)
output1_7 = Dense(10,name='output1_7', activation='relu')(output1_6)
output1_8 = Dense(3,name='output1_8')(output1_7)

# 2
output2_1 = Dense(10,name='output2_1', activation='relu')(middle1_10)
output2_2 = Dense(10,name='output2_2', activation='relu')(output2_1)
output2_3 = Dense(10,name='output2_3', activation='relu')(output2_2)
output2_4 = Dense(10,name='output2_4', activation='relu')(output2_3)
output2_5 = Dense(10,name='output2_5', activation='relu')(output2_4)
output2_6 = Dense(10,name='output2_6', activation='relu')(output2_5)
output2_7 = Dense(10,name='output2_7', activation='relu')(output2_6)
output2_8 = Dense(3,name='output2_8')(output2_7)

# 3
output3_1 = Dense(10,name='output3_1', activation='relu')(middle1_10)
output3_2 = Dense(10,name='output3_2', activation='relu')(output3_1)
output3_3 = Dense(10,name='output3_3', activation='relu')(output3_2)
output3_4 = Dense(10,name='output3_4', activation='relu')(output3_3)
output3_5 = Dense(10,name='output3_5', activation='relu')(output3_4)
output3_6 = Dense(10,name='output3_6', activation='relu')(output3_5)
output3_7 = Dense(10,name='output3_7', activation='relu')(output3_6)
output3_8 = Dense(3,name='output3_8')(output3_7)


# 모델 정의
# model1 = Model(inputs=input1,outputs=output1,name="Model1")
# model2 = Model(inputs=inputt1,outputs=outputt1,name="Model2")
# model1.summary()
# model2.summary()

model = Model(inputs=[input1],outputs=[output1_8,output2_8,output3_8])

model.summary()




# 3. 컴파일 훈련

# from sklearn.model_selection import train_test_split
# x_train, x_test , y_train  , y_test = train_test_split(x , y , train_size=0.6, shuffle=True)
# x_val, x_test , y_val  , y_test = train_test_split(x , y , test_size=0.5, shuffle=True)


model.compile(loss='mse', optimizer='adam', metrics='mse')
model.fit([x1_train],[y1_train,y2_train,y3_train], epochs=1, 
        #    batch_size=1,validation_data=([x1_val,x2_val],[y1_val,y1_val]))
        batch_size=1,validation_data=([x1_val,x1_val,x1_val],[y1_val,y2_val,y3_val]))


# 4. 평가, 예측
result = model.evaluate(([x1_test]),([y1_test]),batch_size=1)
print("result :",result)

y1_test_predict = model.predict(x1_test)
print("y1_test_predict :",y1_test_predict)
# print("y2_test_predict :",y2_test_predict)
# print("y3_test_predict :",y3_test_predict)



# # RMSE 
# from sklearn.metrics import mean_squared_error
# def RMSE(y_test_R,y_test_predict_R):
#         y_t     = y_test_R
#         y_t_pre = y_test_predict_R
#         return np.sqrt(mean_squared_error(y_t,y_t_pre))
# print("RMSE y1 : ",RMSE(y1_test,y1_test_predict)) 
# print("RMSE y2 : ",RMSE(y2_test,y2_test_predict)) 
# print("RMSE y3 : ",RMSE(y3_test,y3_test_predict)) 

# # R2
# from sklearn.metrics import r2_score
# # r2 = r2_score(y_test,x_test_predict)

# print("R2 y1 : ",r2_score(y1_test,y1_test_predict))
# print("R2 y2 : ",r2_score(y2_test,y2_test_predict))
# print("R2 y3 : ",r2_score(y3_test,y3_test_predict))


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








