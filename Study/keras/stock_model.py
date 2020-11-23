import numpy as np

samsung_x = np.load("./data/samsung_x.npy")
bit_x     = np.load("./data/bit_x.npy")
gold_x    = np.load("./data/gold_x.npy")
kosdaq_x  = np.load("./data/kosdaq_x.npy")
samsung_y = np.load("./data/samsung_y.npy")
size = 5

# predict
# samsung
samsung_pred_1 = samsung_x[-3:]
samsung_pred_2 = np.array([[65700,65110,64800],[64301 ,64100 ,64600]])
# samsung_pred_2 = samsung_pred_2.reshape(1,samsung_pred_2.shape[0])
samsung_x_pred =  np.concatenate((samsung_pred_1,samsung_pred_2))

# bit
bit_pred_1 = bit_x[-3:]
bit_pred_2 = np.array([[9839,9620,10000,9880],[12035 ,12800 ,10400 ,12800]])
# bit_pred_2 = bit_pred_2.reshape(1,bit_pred_2.shape[0])
bit_x_pred =  np.concatenate((bit_pred_1,bit_pred_2))

# gold
gold_pred_1 = gold_x[-3:]
gold_pred_2 = np.array([[67176,67290,67360,66980,66990],[67041 ,67160 ,66830 ,66830 ,67000 ]])
# gold_pred_2 = gold_pred_2.reshape(1,gold_pred_2.shape[0])
gold_x_pred =  np.concatenate((gold_pred_1,gold_pred_2))

# kosdaq
kosdaq_pred_1 = kosdaq_x[-3:]
kosdaq_pred_2 = np.array([[6275,842,852,840,849,851],[6578,860 ,849 ,851 ,859 ,630]])
# kosdaq_pred_2 = kosdaq_pred_2.reshape(1,kosdaq_pred_2.shape[0])
kosdaq_x_pred =  np.concatenate((kosdaq_pred_1,kosdaq_pred_2))

# 데이터 전처리
# samsung
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(samsung_x)

# bit
from sklearn.preprocessing import MinMaxScaler
scaler2 = MinMaxScaler()
scaler2.fit(bit_x)

# gold
from sklearn.preprocessing import MinMaxScaler
scaler3 = MinMaxScaler()
scaler3.fit(gold_x)

# kosdaq
from sklearn.preprocessing import MinMaxScaler
scaler4 = MinMaxScaler()
scaler4.fit(kosdaq_x)


# transform
# samsung
samsung_x = scaler.transform(samsung_x)
samsung_x_pred    = scaler.transform(samsung_x_pred)
samsung_x_pred    = samsung_x_pred.reshape(1,samsung_x_pred.shape[0],samsung_x_pred.shape[1])

# bit
bit_x = scaler2.transform(bit_x)
bit_x_pred    = scaler2.transform(bit_x_pred)
bit_x_pred    = bit_x_pred.reshape(1,bit_x_pred.shape[0],bit_x_pred.shape[1])

# gold
gold_x = scaler3.transform(gold_x)
gold_x_pred    = scaler3.transform(gold_x_pred)
gold_x_pred    = gold_x_pred.reshape(1,gold_x_pred.shape[0],gold_x_pred.shape[1])

# kosdaq
kosdaq_x = scaler4.transform(kosdaq_x)
kosdaq_x_pred    = scaler4.transform(kosdaq_x_pred)
kosdaq_x_pred    = kosdaq_x_pred.reshape(1,kosdaq_x_pred.shape[0],kosdaq_x_pred.shape[1])

def new_split (r_data,r_size):
    new_data = []
    for i in range(len(r_data)-(r_size-1)) : 
        new_data.append(r_data[i:i+size])
    return np.array(new_data)

samsung_x = new_split(samsung_x,size)
bit_x     = new_split(bit_x,size)
gold_x    = new_split(gold_x,size)
kosdaq_x  = new_split(kosdaq_x,size)


samsung_y = samsung_y.reshape(samsung_y.shape[0],1)

from sklearn.model_selection import train_test_split
samsung_x_train, samsung_x_test, bit_x_train, bit_x_test, gold_x_train, gold_x_test, kosdaq_x_train, kosdaq_x_test, samsung_y_train , samsung_y_test = train_test_split(samsung_x, bit_x, gold_x, kosdaq_x , samsung_y , train_size=0.8, random_state=1)


# 2. Model
from tensorflow.keras.models import  Model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.layers import Dropout
from keras.layers import Input, Concatenate

# samsung
input1   = Input(shape=(5,3))
LSTM1_1  = LSTM(10,activation='relu',name='LSTM1_1')(input1)
# dense1_1 = Dense(10,activation='relu',name='dense1_1')(LSTM1_1)
# dense1_2 = Dense(10,activation='relu',name='dense1_2')(dense1_1)
# dense1_3 = Dense(10,activation='relu',name='dense1_3')(dense1_2)
dense1_4 = Dense(10,activation='relu',name='dense1_4')(LSTM1_1)


# bit
input2   = Input(shape=(5,4))
LSTM2_1  = LSTM(10,activation='relu',name='LSTM2_1')(input2)
# dense2_1 = Dense(10,activation='relu',name='dense2_1')(LSTM2_1)
# dense2_2 = Dense(10,activation='relu',name='dense2_2')(dense2_1)
# dense2_3 = Dense(10,activation='relu',name='dense2_3')(dense2_2)
dense2_4 = Dense(10,activation='relu',name='dense2_4')(LSTM2_1)


# bit
input3   = Input(shape=(5,5))
LSTM3_1  = LSTM(10,activation='relu',name='LSTM3_1')(input3)
# dense3_1 = Dense(10,activation='relu',name='dense3_1')(LSTM3_1)
# dense3_2 = Dense(10,activation='relu',name='dense3_2')(dense3_1)
# dense3_3 = Dense(10,activation='relu',name='dense3_3')(dense3_2)
dense3_4 = Dense(10,activation='relu',name='dense3_4')(LSTM3_1)


# kosdaq
input4   = Input(shape=(5,6))
LSTM4_1  = LSTM(10,activation='relu',name='LSTM4_1')(input4)
# dense4_1 = Dense(10,activation='relu',name='dense4_1')(LSTM4_1)
# dense4_2 = Dense(10,activation='relu',name='dense4_2')(dense4_1)
# dense4_3 = Dense(10,activation='relu',name='dense4_3')(dense4_2)
dense4_4 = Dense(10,activation='relu',name='dense4_4')(LSTM4_1)

merge1 = Concatenate()([dense1_4,dense2_4,dense3_4,dense4_4])
# merge1 = Concatenate()([LSTM1_1,LSTM2_1,LSTM3_1,LSTM4_1])

middle1_1 = Dense(10,activation='relu',name='middle1_1')(merge1)
middle1_2 = Dense(10,activation='relu',name='middle1_2')(middle1_1)
middle1_3 = Dense(10,activation='relu',name='middle1_3')(middle1_2)

output1 = Dense(1,activation='linear')(middle1_3)

# output1 = Dense(1)(merge1)
model = Model(inputs=(input1,input2,input3,input4),outputs=output1)
model.summary()

model.save("./save/stock_model_B_M_1121.h5")

# 3. 컴파일, 훈련

# ES
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=50, mode='auto')

# ModelCheckPoint
from tensorflow.keras.callbacks import ModelCheckpoint
modelpath = "./model/stock_model_MCP_1121-{epoch:04d}-{val_loss:.4f}.hdf5" 
# 2d = 2자리수 정수, 4f = 소수 4째 자리까지
mcp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                      save_best_only=True, mode="auto")

# Compile
model.compile(loss="mse", optimizer="adam", metrics=["mae"])


# fit
hist = model.fit([samsung_x_train,bit_x_train,gold_x_train,kosdaq_x_train],samsung_y_train, 
                  epochs=100000,batch_size=32,verbose=1,
                  validation_split=0.2,callbacks=[es,mcp])

model.save("./save/stock_model_A_M_1121.h5")
model.save_weights("./save/stock_model_W_1121.h5")

# 4. 평가 예측

result = model.evaluate([samsung_x_test,bit_x_test,gold_x_test,kosdaq_x_test],samsung_y_test,batch_size=32)

print("loss : ",result[0])
print("mae : ",result[1])


y_test_predict = model.predict([samsung_x_pred,bit_x_pred,gold_x_pred,kosdaq_x_pred])
print("예측값 :",y_test_predict)


# loss :  3961350.75
# mae :  1481.2152099609375
# 예측값 : [[6.7811256e+09]]