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


from tensorflow.keras.models import load_model
# ==================== LOAD MODEL+WEIGHT    ==================== #
print("# ==================== LOAD MODEL+WEIGHT    ==================== #")
model1 =  load_model("./save/0195_stock_model_A_M_1121.h5")
model1.summary()
result1 = model1.evaluate([samsung_x_test,bit_x_test,gold_x_test,kosdaq_x_test],samsung_y_test,batch_size=32)
print("model1.mse : ",result1[0])
print("model1.mae : ",result1[1],"\n")

y_test_predict = model1.predict([samsung_x_pred,bit_x_pred,gold_x_pred,kosdaq_x_pred])
print("예측값 :",y_test_predict)


# ==================== LOAD WEIGHT          ==================== #
print("# ==================== LOAD WEIGHT    ==================== #")
model2 =  load_model("./save/0195_stock_model_B_M_1121.h5")
model2.compile(loss="mse", optimizer="adam", metrics=["mae"])
model2.load_weights("./save/0195_stock_model_W_1121.h5")

result2 = model2.evaluate([samsung_x_test,bit_x_test,gold_x_test,kosdaq_x_test],samsung_y_test,batch_size=32)
print("model2.mse : ",result2[0])
print("model2.mae : ",result2[1],"\n")

y_test_predict = model2.predict([samsung_x_pred,bit_x_pred,gold_x_pred,kosdaq_x_pred])
print("예측값 :",y_test_predict)

# ==================== LOAD MODELCHECKPOINT ==================== #
print("# ==================== LOAD MODELCHECKPOINT    ==================== #")
model3 =  load_model("./model/0195_stock_model_MCP_1121-0195-4107265.5000.hdf5")

result3 = model3.evaluate([samsung_x_test,bit_x_test,gold_x_test,kosdaq_x_test],samsung_y_test,batch_size=32)
print("model3.mse : ",result3[0])
print("model3.mae : ",result3[1],"\n")

y_test_predict = model3.predict([samsung_x_pred,bit_x_pred,gold_x_pred,kosdaq_x_pred])
print("예측값 :",y_test_predict)




