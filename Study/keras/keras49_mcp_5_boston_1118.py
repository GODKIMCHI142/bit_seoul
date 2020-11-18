# OneHotEncoding
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
dataset = load_boston()
x = dataset.data
y = dataset.target

# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)

# fit한 결과로 transform
x = scaler.transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test , y_train  , y_test = train_test_split(x , y , train_size=0.8, random_state=1)

# from keras.utils import np_utils
# from tensorflow.keras.utils import to_categorical
# from sklearn.preprocessing import OneHotEncoder

# y_train = to_categorical(y_train)
# y_test  = to_categorical(y_test)

# x_train = x_train.astype('float32')/255.
# x_test  = x_test.astype('float32')/255. 

# 2. Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout

model = Sequential()
model.add(Dense(200,input_shape=(13,),activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(150,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(80,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(50,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))
model.summary()            
model.save("./save/keras49_mcp_5_B_M_1118.h5")

# 3. 컴파일, 훈련

# ES
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=100, mode='auto')

# ModelCheckPoint
from tensorflow.keras.callbacks import ModelCheckpoint
modelpath = "./model/keras49_mcp_5_MCP_1118-{epoch:02d}-{val_loss:.4f}.hdf5" 
# 2d = 2자리수 정수, 4f = 소수 4째 자리까지
mcp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                      save_best_only=True, mode="auto")

# Compile
model.compile(loss="mse", optimizer="adam", metrics=["mae"])

# fit
hist = model.fit(x_train,y_train, epochs=10000,batch_size=32,verbose=1,
          validation_split=0.2,callbacks=[es,mcp])

model.save("./save/keras49_mcp_5_A_M_1118.h5")
model.save_weights("./save/keras49_mcp_5_W_1118.h5")



loss     = hist.history["loss"]
val_loss = hist.history["val_loss"]
mae      = hist.history["mae"]
val_mae  = hist.history["val_mae"]

# 4. 평가, 예측
result = model.evaluate(x_test,y_test,batch_size=10)

print("loss : ",result[0])
print("mae : ",result[1])
# predict
x_pred = x_test[0:10]
y_pred = y_test[0:10]

y_test_predict = model.predict([x_pred])

print("예측값 :",y_test_predict)
print("실제값 :",y_pred)

# RMSE 
from sklearn.metrics import mean_squared_error
def RMSE(y_test_R,y_test_predict_R):
        y_t     = y_test_R
        y_t_pre = y_test_predict_R
        return np.sqrt(mean_squared_error(y_t,y_t_pre))
print("RMSE : ",RMSE(y_pred,y_test_predict)) 

# R2
from sklearn.metrics import r2_score
# r2 = r2_score(y_test,x_test_predict)
print("R2 : ",r2_score(y_pred,y_test_predict))




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
plt.plot(mae,marker='.',c='red')
plt.plot(val_mae,marker='.',c='blue')
plt.grid() # 모눈종이 모양으로 하겠다.

plt.title('mae')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['mae','val_mae']) # 라벨의 위치를 명시해주지 않으면 알아서 빈곳에 노출한다.

plt.show()


# loss :  17.094850540161133
# mae :  2.733506202697754



