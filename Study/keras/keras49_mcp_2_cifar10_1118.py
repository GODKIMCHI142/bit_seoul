# OneHotEncoding
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape,x_test.shape) # (50000, 32, 32,3) (10000, 32, 32,3)
print(y_train.shape,y_test.shape) # (50000,1) (10000,1)


from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)



x_train = x_train.reshape(50000, 32, 32, 3).astype('float32')/255.
x_test  = x_test.reshape(10000, 32, 32, 3).astype('float32')/255. # minmax scaler의 효과



# print(x_train[0]) # 최대값은 255이다. 명암을 0 ~ 255 값으로 나타내었다.

# 2. Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten

model = Sequential()
model.add(Conv2D(60,(2,2),padding='same',input_shape=(32,32,3))) 
model.add(Conv2D(50,(2,2),padding='valid'))                      
model.add(Conv2D(40,(3,3)))                                      
model.add(Conv2D(30,(2,2),strides=2))                            
model.add(MaxPooling2D(pool_size=2))   
model.add(Flatten())                                             
model.add(Dense(20,activation='relu'))                           
model.add(Dense(10,activation='softmax'))                        
model.summary()
model.save("./save/keras49_mcp_2_B_M_1118.h5")
# 3. 컴파일, 훈련

# ES
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=5, mode='auto')

# ModelCheckPoint
from tensorflow.keras.callbacks import ModelCheckpoint
modelpath = "./model/keras49_mcp_2_MCP_1118-{epoch:02d}-{val_loss:.4f}.hdf5" 
# 2d = 2자리수 정수, 4f = 소수 4째 자리까지
mcp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                      save_best_only=True, mode="auto")

# Compile
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

# fit
hist = model.fit(x_train,y_train, epochs=10000,batch_size=32,verbose=1,
          validation_split=0.2,callbacks=[es,mcp])


loss     = hist.history["loss"]
val_loss = hist.history["val_loss"]
acc      = hist.history["acc"]
val_acc  = hist.history["val_acc"]

# 4. 평가, 예측
result = model.evaluate(x_test,y_test,batch_size=32)
print("loss : \n",result[0])
print("accuracy : \n",result[1])

model.save("./save/keras49_mcp_2_A_M_1118.h5")
model.save_weights("./save/keras49_mcp_2_W_1118.h5")

# predict data
x_pred = x_test[0:10]
y_pred = y_test[0:10]

# Y_class_recovery = np.argmax(Y_class_onehot, axis=1).reshape(-1,1)
y_test_predict = model.predict([x_pred])
ytp_recovery   = np.argmax(y_test_predict,axis=1)
y_real         = np.argmax(y_pred,axis=1)

print("예측값 : ",ytp_recovery)
print("실제값 : ",y_real)

# 시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6)) # 단위 무엇인지 찾아볼것
plt.subplot(2,1,1)         # 2행 1열 중 첫번째
plt.plot(loss,marker='.',c='red',label='loss')
plt.plot(val_loss,marker='.',c='blue',label='val_loss')
plt.grid() # 모눈종이 모양으로 하겠다.

plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2,1,2)         # 2행 1열 중 두번째
plt.plot(acc,marker='.',c='red')
plt.plot(val_acc,marker='.',c='blue')
plt.grid() # 모눈종이 모양으로 하겠다.

plt.title('accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc','val_acc']) # 라벨의 위치를 명시해주지 않으면 알아서 빈곳에 노출한다.

plt.show()



# loss : 3.377864122390747
# accuracy : 0.5586000084877014


