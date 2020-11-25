# 회귀모델
import numpy as np

# 1. 데이터 준비
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA

dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x.shape) # (442,10)
print(y.shape) # (442,)

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
model.add(Dense(30,activation='relu',input_shape=(x_train.shape[1],)))
model.add(Dense(50,activation='relu'))
model.add(Dense(70,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

model.summary()

# 3. 컴파일 훈련

# ES
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=100, mode='auto')

model.compile(loss="mse",optimizer="adam",metrics="mae")
model.fit(x_train,y_train,epochs=10000,batch_size=10,validation_split=0.2,callbacks=[es],verbose=1)

print("fit end")
# 4. 평가 예측
loss = model.evaluate(x_test,y_test,batch_size=10)

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

# 0.95 이상
# loss :  [5554.22802734375, 53.440208435058594]
# 예측값 : [[ 52.034378]
#  [149.98248 ]
#  [178.76985 ]
#  [ 83.40051 ]
#  [129.88121 ]
#  [170.40233 ]
#  [236.94223 ]
#  [ 71.09379 ]
#  [165.72209 ]
#  [ 90.30077 ]]
# 실제값 : [ 78. 152. 200.  59. 311. 178. 332. 132. 156. 135.]
# RMSE :  70.302357601276
# R2 :  0.3034339827722867


# 1 이상
# loss :  [4947.30224609375, 54.22142028808594]
# 예측값 : [[121.23579 ]
#  [ 71.36022 ]
#  [123.95814 ]
#  [122.31061 ]
#  [257.98282 ]
#  [178.39032 ]
#  [193.2923  ]
#  [157.16803 ]
#  [125.38896 ]
#  [ 84.660866]]
# 실제값 : [ 78. 152. 200.  59. 311. 178. 332. 132. 156. 135.]
# RMSE :  66.57176536049703
# R2 :  0.3753990335705657