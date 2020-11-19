import numpy as np

x = np.load("./data/iris_x.npy")
y = np.load("./data/iris_y.npy")

# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)

# fit한 결과로 transform
x = scaler.transform(x)


from sklearn.model_selection import train_test_split
x_train, x_test , y_train  , y_test = train_test_split(x , y , train_size=0.8)
x_train = x_train.reshape(120,4,1,1)
x_test  = x_test.reshape(30,4,1,1)
y_train = y_train.reshape(120,1)
y_test = y_test.reshape(30,1)

from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)

from tensorflow.keras.models import load_model
# ==================== LOAD MODEL+WEIGHT    ==================== #
print("# ==================== LOAD MODEL+WEIGHT    ==================== #")
model1 =  load_model("./save/keras49_mcp_7_A_M_1118.h5")

result1 = model1.evaluate(x_test,y_test,batch_size=32)
print("model1.loss : ",result1[0])
print("model1.accuracy : ",result1[1],"\n")

# ==================== LOAD WEIGHT          ==================== #
print("# ==================== LOAD WEIGHT    ==================== #")
model2 =  load_model("./save/keras49_mcp_7_B_M_1118.h5")
model2.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
model2.load_weights("./save/keras49_mcp_7_W_1118.h5")

result2 = model2.evaluate(x_test,y_test,batch_size=32)
print("model2.loss : ",result2[0])
print("model2.accuracy : ",result2[1],"\n")

# ==================== LOAD MODELCHECKPOINT ==================== #
print("# ==================== LOAD MODELCHECKPOINT    ==================== #")
model3 =  load_model("./model/keras49_mcp_7_MCP_1118-191-0.0836.hdf5")

result3 = model3.evaluate(x_test,y_test,batch_size=32)
print("model3.loss : ",result3[0])
print("model3.accuracy : ",result3[1],"\n")