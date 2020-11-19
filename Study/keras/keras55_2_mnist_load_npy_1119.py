import numpy as np
import matplotlib.pyplot as plt
# from tensorflow.keras.datasets import mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape,x_test.shape) # (60000, 28, 28) (10000, 28, 28)
# print(y_train.shape,y_test.shape) # (60000,) (10000,)

# x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
# x_test  = x_test.reshape(10000, 28, 28, 1).astype('float32')/255. 
# from keras.utils import np_utils
# from tensorflow.keras.utils import to_categorical
# from sklearn.preprocessing import OneHotEncoder

# y_train = to_categorical(y_train)
# y_test  = to_categorical(y_test)



x_train = np.load("./data/mnist_x_train.npy")
x_test  = np.load("./data/mnist_x_test.npy")
y_train = np.load("./data/mnist_y_train.npy")
y_test  = np.load("./data/mnist_y_test.npy")

print(x_train.shape,x_test.shape) # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape,y_test.shape) # (60000,) (10000,)



print(x_train.shape,x_test.shape) # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape,y_test.shape) # (60000,) (10000,)

from tensorflow.keras.models import load_model
# ==================== LOAD MODEL+WEIGHT    ==================== #
print("# ==================== LOAD MODEL+WEIGHT    ==================== #")
model1 =  load_model("./save/keras49_mcp_1_A_M_1118.h5")

result1 = model1.evaluate(x_test,y_test,batch_size=32)
print("model1.loss : ",result1[0])
print("model1.accuracy : ",result1[1],"\n")

# ==================== LOAD WEIGHT          ==================== #
print("# ==================== LOAD WEIGHT    ==================== #")
model2 =  load_model("./save/keras49_mcp_1_B_M_1118.h5")
model2.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
model2.load_weights("./save/keras49_mcp_1_W_1118.h5")

result2 = model2.evaluate(x_test,y_test,batch_size=32)
print("model2.loss : ",result2[0])
print("model2.accuracy : ",result2[1],"\n")

# ==================== LOAD MODELCHECKPOINT ==================== #
print("# ==================== LOAD MODELCHECKPOINT    ==================== #")
model3 =  load_model("./model/keras49_mcp_1_MCP_1118-06-0.0602.hdf5")

result3 = model3.evaluate(x_test,y_test,batch_size=32)
print("model3.loss : ",result3[0])
print("model3.accuracy : ",result3[1],"\n")



