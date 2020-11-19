import numpy as np

x = np.load("./data/diabetes_x.npy")
y = np.load("./data/diabetes_y.npy")


# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)

# fit한 결과로 transform
x = scaler.transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test , y_train  , y_test = train_test_split(x , y , train_size=0.8, random_state=1)


from tensorflow.keras.models import load_model
# ==================== LOAD MODEL+WEIGHT    ==================== #
print("# ==================== LOAD MODEL+WEIGHT    ==================== #")
model1 =  load_model("./save/keras49_mcp_6_A_M_1118.h5")

result1 = model1.evaluate(x_test,y_test,batch_size=10)
print("model1.mse : ",result1[0])
print("model1.mae : ",result1[1],"\n")

# ==================== LOAD WEIGHT          ==================== #
print("# ==================== LOAD WEIGHT    ==================== #")
model2 =  load_model("./save/keras49_mcp_6_B_M_1118.h5")
model2.compile(loss="mse", optimizer="adam", metrics=["mae"])
model2.load_weights("./save/keras49_mcp_6_W_1118.h5")

result2 = model2.evaluate(x_test,y_test,batch_size=10)
print("model2.mse : ",result2[0])
print("model2.mae : ",result2[1],"\n")

# ==================== LOAD MODELCHECKPOINT ==================== #
print("# ==================== LOAD MODELCHECKPOINT    ==================== #")
model3 =  load_model("./model/keras49_mcp_6_MCP_1118-35-2931.1433.hdf5")

result3 = model3.evaluate(x_test,y_test,batch_size=10)
print("model3.mse : ",result3[0])
print("model3.mae : ",result3[1],"\n")