import numpy as np

x = np.load("./data/samsung_x.npy")
y = np.load("./data/samsung_y.npy")
size = 5

# print(x.shape) # [59723 60300]
# print(y.shape)
# print(x)
# print(x[619])
# print(x[620])


# predict
x_pred = np.array([[[60955, 61300],
[61028, 61000],
[65589, 66300],
[66194, 65700],
[65110 ,64800]]])



# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)

# fit한 결과로 transform
x = scaler.transform(x)
x_pred = scaler.transform(x_pred.reshape(5,2))
x_pred = x_pred.reshape(1,5,2)
# (150,4)를 받아서 size행씩 잘라서 다시 만든다.

def new_split (r_data,r_size):
    new_data = []
    for i in range(len(r_data)-(r_size-1)) : 
        new_data.append(r_data[i:i+size])
    return np.array(new_data)

x = new_split(x,size)
print(x.shape)
print(type(x))
# print(x)


y = y.reshape(y.shape[0],1)

from sklearn.model_selection import train_test_split
x_train, x_test , y_train  , y_test = train_test_split(x , y , train_size=0.8, random_state=1)

# 2. Model
from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from keras.layers import Input, concatenate

input1 = Input(shape=(5,2))
LSTM1_1 = LSTM(50,activation='relu',name='LSTM1_1')(input1)
dense1_1 = Dense(60,activation='relu',name='dense1_1')(LSTM1_1)
dense1_2 = Dense(70,activation='relu',name='dense1_2')(dense1_1)
dense1_3 = Dense(80,activation='relu',name='dense1_3')(dense1_2)
dense1_4 = Dense(70,activation='relu',name='dense1_4')(dense1_3)
dense1_5 = Dense(60,activation='relu',name='dense1_5')(dense1_4)
dense1_6 = Dense(50,activation='relu',name='dense1_6')(dense1_5)
dense1_7 = Dense(30,activation='relu',name='dense1_7')(dense1_6)
dense1_8 = Dense(20,activation='relu',name='dense1_8')(dense1_7)
dense1_9 = Dense(10,activation='relu',name='dense1_9')(dense1_8)
output1 = Dense(1,name="output2")(dense1_9)

model = Model(inputs=(input1),outputs=output1)
model.summary()


model.save("./save/l4_B_M_1118.h5")

# 3. 컴파일, 훈련

# ES
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=100, mode='auto')

# ModelCheckPoint
from tensorflow.keras.callbacks import ModelCheckpoint
modelpath = "./model/l4_MCP_1118-{epoch:02d}-{val_loss:.4f}.hdf5" 
# 2d = 2자리수 정수, 4f = 소수 4째 자리까지
mcp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                      save_best_only=True, mode="auto")

# Compile
model.compile(loss="mse", optimizer="adam", metrics=["mae"])

# fit
hist = model.fit(x_train,y_train, epochs=10000,batch_size=10,verbose=1,
          validation_split=0.2,callbacks=[es,mcp])

model.save("./save/l4_A_M_1118.h5")
model.save_weights("./save/l4_W_1118.h5")



loss     = hist.history["loss"]
val_loss = hist.history["val_loss"]
mae      = hist.history["mae"]
val_mae  = hist.history["val_mae"]

# 4. 평가, 예측
result = model.evaluate(x_test,y_test,batch_size=10)

print("loss : ",result[0])
print("mae : ",result[1])


y_test_predict = model.predict([x_pred])
y_real = np.array([[63900]])
print("예측값 :",y_test_predict)
print("실제값 :",y_real)

# RMSE 
from sklearn.metrics import mean_squared_error
def RMSE(y_test_R,y_test_predict_R):
        y_t     = y_test_R
        y_t_pre = y_test_predict_R
        return np.sqrt(mean_squared_error(y_t,y_t_pre))
print("RMSE : ",RMSE(y_real,y_test_predict)) 






