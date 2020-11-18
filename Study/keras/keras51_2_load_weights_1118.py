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


# 2. Model
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.layers import MaxPooling2D
# from tensorflow.keras.layers import Flatten
# from tensorflow.keras.models import load_model
# model = load_model("./save/keras51_1_save_weight_1118_model_test02_2.h5")
# model.summary()                     

from tensorflow.keras.models import load_model
model = load_model("./model/keras50_2_load1_model_1118-25-0.3268.hdf5")

# 3. 컴파일, 훈련

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



# loss : 0.3558720350265503
# accuracy : 0.8784999847412109

