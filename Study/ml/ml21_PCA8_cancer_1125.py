# 이진분류

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
print(x.shape) 
print(y.shape) 

# cumsum
pca = PCA()
pca.fit(x) # 중요도가 높은순서대로 바뀐다.
cumsum = np.cumsum(pca.explained_variance_ratio_) 

d = np.argmax(cumsum) + 1

pca = PCA(n_components=d)
x = pca.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test , y_train  , y_test = train_test_split(x , y , train_size=0.8, random_state=1)

# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)

# fit한 결과로 transform
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D
from tensorflow.keras.layers import Dropout

model = Sequential()
model.add(Dense(50,activation='relu',input_shape=(x_train.shape[1],)))
model.add(Dense(40,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.summary()

# 3. 컴파일 훈련

# ES
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=100, mode='auto')

model.compile(loss="binary_crossentropy",optimizer="adam",metrics="acc")
model.fit(x_train,y_train,epochs=1000,batch_size=8,validation_split=0.3,callbacks=[es],verbose=1)

print("fit end")
# 4. 평가 예측
loss, acc = model.evaluate(x_test,y_test,batch_size=8)

print("loss : ",loss)
print("acc : ",acc)
# predict
x_pred = x_test[0:10]
y_pred = y_test[0:10]

y_test_predict = model.predict(x_pred)

print("예측값 :",y_test_predict)
print("실제값 :",y_pred)

# 32/32 [==============================] - 0s 2ms/step - loss: 7.8383e-04 - acc: 1.0000 - val_loss: 0.2062 - val_acc: 0.9708
# fit end
#  1/12 [=>............................] - ETA: 0s - loss: 4.9902e-06 - acc: 1.0000WARNING:tensorflow:Callbacks method `on_test_batch_end` is slow compared to the batch time (batch time: 0.0000s vs `on_test_batch_end` time: 0.0010s). Check your callbacks.
# 12/12 [==============================] - 0s 1ms/step - loss: 0.0814 - acc: 0.9825
# loss :  0.0814206525683403
# acc :  0.9824561476707458
# 예측값 : [[1.7883267e-10]
#  [9.9999964e-01]
#  [1.0000000e+00]
#  [9.9999964e-01]
#  [1.0000000e+00]
#  [9.9995363e-01]
#  [9.9999726e-01]
#  [4.7543941e-10]
#  [5.2713101e-08]
#  [1.1464765e-20]]
# 실제값 : [0 1 1 1 1 1 1 0 0 0]

# 0.95 이상
# loss :  0.32116997241973877
# acc :  0.8771929740905762
# 예측값 : [[7.05219328e-01]
#  [7.76298583e-01]
#  [9.74086285e-01]
#  [1.87821202e-02]
#  [4.36098605e-01]
#  [1.15610495e-01]
#  [7.21909048e-04]
#  [5.84796108e-02]
#  [9.99599397e-01]
#  [7.42506146e-01]]
# 실제값 : [1 0 1 0 0 0 0 0 1 1]

# 1 이상
# loss :  0.5055868029594421
# acc :  0.9298245906829834
# 예측값 : [[9.9997938e-01]
#  [2.7394913e-08]
#  [9.9999702e-01]
#  [5.2365465e-07]
#  [2.5491154e-06]
#  [3.8659845e-08]
#  [2.9136304e-13]
#  [9.9997675e-01]
#  [9.9999213e-01]
#  [9.9999380e-01]]
# 실제값 : [1 0 1 0 0 0 0 0 1 1]
