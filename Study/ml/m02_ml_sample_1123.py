import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC

dataset = load_iris()
x , y = load_iris(return_X_y=True)
# x = dataset.data
# y = dataset.target
# print(x.shape) # 150, 4
# print(y.shape) # 150,



# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)

# fit한 결과로 transform
x = scaler.transform(x)


from sklearn.model_selection import train_test_split
x_train, x_test , y_train  , y_test = train_test_split(x , y , train_size=0.8,shuffle=True)


# one hot encoding
# from keras.utils import np_utils
# from tensorflow.keras.utils import to_categorical
# from sklearn.preprocessing import OneHotEncoder

# y_train = to_categorical(y_train)
# y_test  = to_categorical(y_test)

# 2. 모델
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.layers import MaxPooling2D
# from tensorflow.keras.layers import Flatten
# from tensorflow.keras.layers import Dropout

# model = Sequential()
# model.add(Dense(30,input_shape=(4,),activation='relu')) 
# model.add(Dense(20,activation='relu'))  
# model.add(Dense(10,activation='relu'))
# model.add(Dense(3,activation='softmax'))                       

model = LinearSVC()

# 3. 컴파일 훈련

# ES
# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='loss',patience=100, mode='auto')

# model.compile(loss="mse",optimizer="adam",metrics="acc")
# model.fit(x_train,y_train,epochs=10000,batch_size=10,validation_split=0.2,verbose=1)
model.fit(x_train,y_train)

# 4. 평가 예측
# loss , acc= model.evaluate(x_test,y_test,batch_size=10)

# print("loss : ",loss)
# print("acc : ",acc)

result = model.score(x_test,y_test)# evaluate 대신 시ㅏ용
print("result : ",result)

# predict
x_pred = x_test[0:10]
y_pred = y_test[0:10]

# Y_class_recovery = np.argmax(Y_class_onehot, axis=1).reshape(-1,1)
# y_test_predict = model.predict([x_pred])

# ytp_recovery = np.argmax(y_test_predict,axis=1)
# print("예측값 :",ytp_recovery)

# y_real = np.argmax(y_pred,axis=1)
# print("실제값 :",y_real)


# loss :  0.04196592792868614
# acc :  0.9333333373069763
# 예측값 : [2 1 0 2 0 1 1 1 0 0]
# 실제값 : [2 1 0 2 0 1 1 1 0 0]










