# OneHotEncoding
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)


from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)



x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test  = x_test.reshape(10000,28,28,1).astype('float32')/255. # minmax scaler의 효과

# print(x_train[0]) # 최대값은 255이다. 명암을 0 ~ 255 값으로 나타내었다.

# 2. Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten

model = Sequential()
model.add(Conv2D(10,(2,2),padding='same',input_shape=(28,28,1))) # 28,28,60
model.add(Conv2D(10,(2,2),padding='valid'))                      # 27,27,50
model.add(Conv2D(10,(3,3)))                                      # 25,25,40
model.add(Conv2D(10,(2,2),strides=2))                            # 12,12,30
model.add(MaxPooling2D(pool_size=2))   # pool_size default : 2   # 6 ,6 ,30
model.add(Flatten())                                             # 1080,
model.add(Dense(10,activation='relu'))                           # 20,
model.add(Dense(10,activation='softmax'))                        # 10,
model.summary()                     

# model save
# model.save("./save/keras51_1_save_weight_1118_test01_1.h5")


# 3. 컴파일, 훈련

# ES
# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='loss',patience=5, mode='auto')

# # ModelCheckPoint
# from tensorflow.keras.callbacks import ModelCheckpoint
# modelpath = "./model/keras51_1_save_weight_1118-{epoch:02d}-{val_loss:.4f}.hdf5" 
# # 2d = 2자리수 정수, 4f = 소수 4째 자리까지
# mcp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
#                       save_best_only=True, mode="auto")

# Compile
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

# # fit
# hist = model.fit(x_train,y_train, epochs=30,batch_size=32,verbose=1,
#           validation_split=0.2,callbacks=[es,mcp])

model.load_weights("./save/keras51_1_save_weight_1118_weight_test02.h5")

# 4. 평가, 예측
result = model.evaluate(x_test,y_test,batch_size=32)
print("loss : ",result[0])
print("accuracy : ",result[1])

# predict data
x_pred = x_test[0:10]
y_pred = y_test[0:10]

# Y_class_recovery = np.argmax(Y_class_onehot, axis=1).reshape(-1,1)
y_test_predict = model.predict([x_pred])
ytp_recovery   = np.argmax(y_test_predict,axis=1)
y_real         = np.argmax(y_pred,axis=1)

print("예측값 : ",ytp_recovery)
print("실제값 : ",y_real)

# loss :  0.3558720350265503
# accuracy :  0.8784999847412109
# 예측값 :  [9 2 1 1 6 1 4 6 5 7]
# 실제값 :  [9 2 1 1 6 1 4 6 5 7]


