import numpy as np

# 1. 데이터 준비
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])


from tensorflow.keras.models import Sequential # tf 안에 keras 안에 model에 seq를 가져온다
from tensorflow.keras.layers import Dense,Activation # tf 안에 keras 안에 layers에 Dense를 가져온다.
from sklearn.metrics import mean_squared_error
from keras.optimizers import SGD

# 2. 모델구성
# DNN을 Dense층으로 구성한다.
# input_dim : 입력 뉴런의 수를 설정합니다.
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
# mse : Mean Squared Error : 손실함수 : 평균 제곱 오차 : 정답에 대한 오류를 숫자로 나타내는 것
# optimizer(최적화)를 adam으로 사용하겠다.
# metrics : 평가지표 
# acc : accuracy : 정확성

model.fit(x,y, epochs=200, batch_size=1)
# model.fit : 이 모델을 훈련시키겠다.
# epochs : 몇번 훈련시키겠다.
# batch_size : 몇 개의 샘플로 가중치를 갱신할 것인지 지정

# 4. 평가, 예측
loss, acc = model.evaluate(x,y,batch_size=1)
print("loss : ",loss)
print("acc : ",acc)









