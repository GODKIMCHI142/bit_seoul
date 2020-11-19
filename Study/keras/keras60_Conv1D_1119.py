import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.keras.layers import MaxPooling1D


# 1. 데이터
a = np.array(range(1,101))
size = 5

def split_x(seq, size):                        # 함수선언
    aaa = []                                   # 빈 리스트를 만듬
    for i in range(len(seq) - size +1):        # for문이 도는 횟수는 dataset의 길이 - 열의 길이 + 1
        subset = seq[i : (i+size)]             # 인덱스 i ~ i+5까지 리스트로 뽑고 1만큼 순차적으로 증가한다
        aaa.append([item for item in subset])  # subset안에서 가져온 데이터를 aaa에 넣는다
    print("type(aaa) : >>> \n",type(aaa))      # aaa의 타입을 출력해본다.                  
    return np.array(aaa)                       # aaa를 numpy.ndarray 타입으로 바꾼다.


datasets = split_x(a, size) # 함수를 호출하고 dataset과 size를 넘겨준뒤 새로운 리스트를 리턴받는다.
print("=================")
print("datasets : >>>> \n",datasets)
print("type(datasets) : >>>> \n",type(datasets))


# Conv1D로 모델을 구성하시오
x = datasets[:,0:4]
y = datasets[:,4]
print(x)
print(y)
x_pred = np.array(range(97,101))
print(x_pred)
print(x_pred.shape)


# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)

# fit한 결과로 transform
x = scaler.transform(x)
x_pred = scaler.transform([x_pred])


x = x.reshape(96,2,2)
y = y.reshape(96,1)
from sklearn.model_selection import train_test_split
x_train, x_test , y_train  , y_test = train_test_split(x , y , train_size=0.8, random_state=1)


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# 모델

model = Sequential()
model.add(Conv1D(10,2,padding='same',input_shape=(2,2),activation='relu'))
model.add(Conv1D(10,2,padding='same',activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

model.summary()

# Compile fit

# ES
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=5, mode='auto')

# Compile
model.compile(loss="mse", optimizer="adam", metrics=["mae"])

# fit
hist = model.fit(x_train,y_train, epochs=10000,batch_size=10,verbose=1,
          validation_split=0.2,callbacks=[es])


# 평가, 예측
result = model.evaluate(x_test,y_test,batch_size=10)

print("loss : ",result[0])
print("mae : ",result[1])

x_pred = x_pred.reshape(1,2,2)
x_predict = model.predict([x_pred])

print("예측값 : ",x_predict)
print("실제값 : ",101)

# loss :  7.409391766799445e-09
# mae :  7.76052474975586e-05
# 예측값 :  [[101.00015]]
# 실제값 :  101


