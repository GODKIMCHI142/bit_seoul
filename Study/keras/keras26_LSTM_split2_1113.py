import numpy as np



# 모델을 구성하시오 fit()까지

# 1. 데이터 준비
dataset = np.array(range(1,101))
size = 5
print("dataset : >>> \n",dataset) # [ 1  2  3  4  5  6  7  8  9 10]
print("len(dataset) : >>> \n",len(dataset)) # 10

def split_x(seq, size):                        
    aaa = []       
    print(type(seq))                            
    for i in range(len(seq) - size +1):        
        subset = seq[i : (i+size)]             
        aaa.append([item for item in subset])  
    print("type(aaa) : >>> \n",type(aaa))                        
    return np.array(aaa)                       


datasets = split_x(dataset, size) 
print("=================")
print("datasets : >>>> \n",datasets)
print("type(datasets) : >>>> \n",type(datasets))

x = []
for i in range(5):
    print("i : ",i)
    print("datasets[i][:5] : >>>",datasets[i][:4])
    x.append(datasets[i][:4])
    print("x >>> : \n",x)

print(datasets[size])
x = np.array(x)
print("x : >>> \n",x)

y = []
for i in range(size):
    print("i : ",i)
    print("datasets[i][:5] : >>>",datasets[i][size-1:])
    y.append(datasets[i][size-1:])
    print("y >>> : \n",y)

print(datasets[size])
y = np.array(y)
print("y : >>> \n",y)

print("x.shape : >>>> ",x.shape)
print("y.shape : >>>> ",y.shape)
x = x.reshape(5,4,1)
# print("x : >>>>> ",x)

# y = y.reshape(1,5,1)
# print("x : >>>>> ",)



from sklearn.model_selection import train_test_split
x_train, x_test , y_train  , y_test = train_test_split(x , y , train_size=0.6)
x_val, x_test , y_val  , y_test = train_test_split(x_test , y_test , test_size=0.5,random_state=1)


# 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(100,activation='relu',input_shape=(3,1)))
model.add(Dense(1500,activation='relu'))
model.add(Dense(1000,activation='relu'))
model.add(Dense(500,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(1))

model.summary()

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor='loss',patience=10, mode='min')
early_stopping = EarlyStopping(monitor='loss',patience=100, mode='auto')

model.compile(loss='mse',optimizer='adam',metrics='mae')
model.fit(x_train,y_train,epochs=10000,batch_size=1,callbacks=[early_stopping],validation_data=(x_val,y_val))



# 4. 평가, 예측

loss , mse = model.evaluate(x_test,y_test)
print("loss : ",loss)
print("mse : ",mse)

x_pred = np.array([97,98,99,100])
x_pred = x_pred.reshape(1,4,1)
x_predict = model.predict(x_pred)
print("x_predict : ",x_predict)

# y_test_predict = model.predict(x_test)
# print("y_test : ",y_test)
# print("y_test_predict : ",y_test_predict)


