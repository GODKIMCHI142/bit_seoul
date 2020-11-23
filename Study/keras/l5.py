import numpy as np

x = np.load("./data/samsung_x.npy")
y = np.load("./data/samsung_y.npy")
size = 5


# predict
x_pred = np.array([[[61028, 61000],
                    [65589 ,66300],
                    [66194 ,65700],
                    [65110 ,64800],
                    [64301 ,64600]]])


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

from tensorflow.keras.models import load_model
# ==================== LOAD MODEL+WEIGHT    ==================== #
print("# ==================== LOAD MODEL+WEIGHT    ==================== #")
model1 =  load_model("./save/461_l4_A_M_1118.h5")

result1 = model1.evaluate(x_test,y_test,batch_size=10)
print("model1.mse : ",result1[0])
print("model1.mae : ",result1[1],"\n")

y_test_predict = model1.predict([x_pred])
print("예측값 :",y_test_predict)

# ==================== LOAD WEIGHT          ==================== #
print("# ==================== LOAD WEIGHT    ==================== #")
model2 =  load_model("./save/461_l4_B_M_1118.h5")
model2.compile(loss="mse", optimizer="adam", metrics=["mae"])
model2.load_weights("./save/461_l4_W_1118.h5")

result2 = model2.evaluate(x_test,y_test,batch_size=10)
print("model2.mse : ",result2[0])
print("model2.mae : ",result2[1],"\n")

y_test_predict = model2.predict([x_pred])
print("예측값 :",y_test_predict)

# ==================== LOAD MODELCHECKPOINT ==================== #
print("# ==================== LOAD MODELCHECKPOINT    ==================== #")
model3 =  load_model("./model/461_l4_MCP_1118-461-594024.3750.hdf5")

result3 = model3.evaluate(x_test,y_test,batch_size=10)
print("model3.mse : ",result3[0])
print("model3.mae : ",result3[1],"\n")

y_test_predict = model3.predict([x_pred])
print("예측값 :",y_test_predict)




