from keras.layers import Dense
from keras.models import Sequential
import numpy as np

x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])
z = np.array([6,7,8,9,10])

model = Sequential()
model.add(Dense(10,input_dim=1))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics='acc')
model.fit(x,y,epochs=100,batch_size=1)

zz = model.predict(z)
loss, acc = model.evaluate(x,y,batch_size=1)
val_loss, val_acc = model.evaluate(z,zz,batch_size=1)

print("loss : ",loss)
print("acc : ",acc)
print("val_loss : ",val_loss)
print("val_acc : ",val_acc)
















