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
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam 

# optimizer = Adam(learning_rate=0.1)
# optimizer = Adadelta(learning_rate=0.1)
# optimizer = Adagrad(learning_rate=0.1)
# optimizer = Adamax(learning_rate=0.1)
# optimizer = RMSprop(learning_rate=0.1)
# optimizer = SGD(learning_rate=0.1)
optimizer = Nadam(learning_rate=0.1)

model.compile(loss='mse', optimizer=optimizer, metrics='mse')


model.fit(x,y, epochs=200, batch_size=1)


# 4. 평가, 예측
loss,mse = model.evaluate(x,y,batch_size=1)
print("loss    : ",loss)

y_predict = model.predict([11])
print("predict : ",y_predict)

# Adam
# lr=0.0001
    # loss    :  2.782542196655413e-09
    # predict :  [[10.99976]]
# lr=0.1
    # loss    :  1642048.5
    # predict :  [[5346.697]]

# Adadelta
# lr=0.0001
    # loss    :  10.263284683227539
    # predict :  [[0.36376145]]
# lr=0.1
    # loss    :  1.2761180414599949e-06
    # predict :  [[10.993747]]

# Adagrad
# lr=0.0001
    # loss    :  0.08078443259000778
    # predict :  [[9.593513]]
# lr=0.1
    # loss    :  5.306483217282221e-06
    # predict :  [[10.98971]]

# Adamax
# lr=0.0001
    # loss    :  0.024199696257710457
    # predict :  [[10.248308]]
# lr=0.1
    # loss    :  5.7355198805453256e-05
    # predict :  [[10.963759]]

# RMSprop
# lr=0.0001
    # loss    :  0.03963180631399155
    # predict :  [[10.30046]]
# lr=0.1
    # loss    :  1.9981505870819092
    # predict :  [[2.9158237]]

# SGD
# lr=0.0001
    # loss    :  1.921100221125016e-07
    # predict :  [[10.999282]]
# lr=0.1
    # loss    :  nan
    # predict :  [[nan]]

# Nadam
# lr=0.0001
    # loss    :  1.2789769243681803e-13
    # predict :  [[10.999997]]
# lr=0.1
    # loss    :  66714.3984375
    # predict :  [[1145.3574]]
