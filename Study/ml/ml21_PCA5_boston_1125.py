# 전처리 트레인 테스트 분리 RMSE R2
import numpy as np


# 1. 데이터 준비
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA

dataset = load_boston()
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
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

model = Sequential()
model.add(Dense(100,activation='relu',input_shape=(x_train.shape[1],)))
model.add(Dense(80,activation='relu'))
model.add(Dense(60,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(20,activation='relu')) 
model.add(Dense(10,activation='relu')) 
model.add(Dense(1))

model.summary()

# 3. 컴파일 훈련

# ES
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=100, mode='auto')

model.compile(loss="mse",optimizer="adam",metrics="mae")
model.fit(x_train,y_train,epochs=1000,batch_size=8,validation_split=0.2,callbacks=[es],verbose=1)

print("fit end")
# 4. 평가 예측
loss = model.evaluate(x_test,y_test,batch_size=8)

print("loss : ",loss)

# predict
x_pred = x_test[0:10]
y_pred = y_test[0:10]

y_test_predict = model.predict([x_pred])

print("예측값 :",y_test_predict)
print("실제값 :",y_pred)

# RMSE 
from sklearn.metrics import mean_squared_error
def RMSE(y_test_R,y_test_predict_R):
        y_t     = y_test_R
        y_t_pre = y_test_predict_R
        return np.sqrt(mean_squared_error(y_t,y_t_pre))
print("RMSE : ",RMSE(y_pred,y_test_predict)) 

# R2
from sklearn.metrics import r2_score
# r2 = r2_score(y_test,x_test_predict)
print("R2 : ",r2_score(y_pred,y_test_predict))




# 11/11 [==============================] - 0s 1ms/step - loss: 12.6104 - mae: 2.5685
# loss :  [12.610381126403809, 2.5684638023376465]
# 예측값 : [[28.618101][22.986748][16.995024][20.615458][20.591936]
#           [19.348417][27.58215 ][16.878729][21.071262][23.160017]]
# 실제값 : [28.2 23.9 16.6 22.  20.8 23.  27.9 14.5 21.5 22.6]
# RMSE :  1.507182859562468
# R2 :  0.8625105815180427



# 0.95 이상
# loss :  [82.4325180053711, 6.1070237159729]
# 예측값 : [[24.837364]
#  [28.796223]
#  [24.286612]
#  [27.142372]
#  [23.697853]
#  [17.981798]
#  [23.49321 ]
#  [21.581465]
#  [22.773993]
#  [26.848192]]
# 실제값 : [28.2 23.9 16.6 22.  20.8 23.  27.9 14.5 21.5 22.6]
# RMSE :  4.935841807410464
# R2 :  -0.4745511649788765


# 1 이상
# loss :  [9.009055137634277, 2.3048949241638184]
# 예측값 : [[29.93743 ]
#  [26.030884]
#  [15.575129]
#  [20.190971]
#  [23.193508]
#  [20.010063]
#  [28.023144]
#  [16.368195]
#  [19.820587]
#  [23.902328]]
# 실제값 : [28.2 23.9 16.6 22.  20.8 23.  27.9 14.5 21.5 22.6]
# RMSE :  1.8593729762908315
# R2 :  0.7907476174215817

