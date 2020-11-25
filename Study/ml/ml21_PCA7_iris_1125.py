# 다중분류
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA


dataset = load_iris()
x = dataset.data
y = dataset.target
print(x.shape) # 150, 4
print(y.shape) # 150,



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


from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)



# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout

model = Sequential()
model.add(Dense(40,activation='relu',input_shape=(x_train.shape[1],)))
model.add(Dense(30,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(3,activation='softmax'))                    
model.summary()

# 3. 컴파일 훈련

# ES
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=100, mode='auto')

model.compile(loss="mse",optimizer="adam",metrics="acc")
model.fit(x_train,y_train,epochs=1000,batch_size=10,validation_split=0.2,callbacks=[es],verbose=1)

print("fit end")
# 4. 평가 예측
loss , acc= model.evaluate(x_test,y_test,batch_size=10)

print("loss : ",loss)
print("acc : ",acc)


# predict
x_pred = x_test[0:10]
y_pred = y_test[0:10]

# Y_class_recovery = np.argmax(Y_class_onehot, axis=1).reshape(-1,1)
y_test_predict = model.predict([x_pred])

ytp_recovery = np.argmax(y_test_predict,axis=1)
print("예측값 :",ytp_recovery)

y_real = np.argmax(y_pred,axis=1)
print("실제값 :",y_real)


# 0.95 이상
# loss :  0.01600600779056549
# acc :  0.9666666388511658
# 예측값 : [0 1 1 0 2 1 2 0 0 2]
# 실제값 : [0 1 1 0 2 1 2 0 0 2]

# 1 이상
# loss :  0.025677591562271118
# acc :  0.9666666388511658
# 예측값 : [0 1 1 0 1 1 2 0 0 2]
# 실제값 : [0 1 1 0 2 1 2 0 0 2]
