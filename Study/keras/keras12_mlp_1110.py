
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
# 1. 데이터
# x1 = np.array([range(1,101),range(311,411),range(100)])
# y1 = np.array([range(101,201),range(711,811),range(100)])
# print("x1.shape : ",x1.shape) # (3,100)

x = np.array((range(1,101),range(311,411),range(100)))
y = np.array((range(101,201),range(711,811),range(100)))
print("x.shape : ",x.shape)  # (3,100)

# (100,3)
print(x)


x_len = x.__len__()
print("x.__len__() : ",x.__len__())

new_x = []
np.array(new_x)
for i in range(x[1,].size):
    # print("s",x[1,].size)
    # new_x += [x[0,i],x[1,i],x[2,i]]
    # on = np.array(x[0,i])
    # tw = np.array(x[1,i])
    # th = np.array(x[2,i])
    # new_x = np.append(new_x,[on],axis=0)
    # new_x = np.append(new_x,[tw],axis=0)
    # new_x = np.append(new_x,[th],axis=0)
    new_x.append([x[0,i],x[1,i],x[2,i]])
new_x = np.array(new_x)
print(new_x)
print("new_x.shape >>> ",new_x.shape)








