# (150,4)
import numpy as np
import pandas as pd

# from sklearn.datasets import load_iris
# s = load_iris()
# dataset = s.data

dataset = pd.read_csv("./data/csv/iris_ys.csv",header=0,index_col=0,sep=",")

# print(dataset)
print(dataset.shape)
print(type(dataset))


# print(dataset)
# print(dataset.shape)
# print(type(dataset))

dataset = dataset.values
size = 5

# (150,5)를 받아서 size행씩 잘라서 다시 만든다.
def new_split (r_data,r_size):
    new_data = []
    for i in range(len(r_data)-(r_size-1)) : 
        new_data.append(r_data[i:i+size])
    # (146,5,5)
    return np.array(new_data)

data2 = new_split(dataset,size)
print(data2.shape)
print(type(data2))


# 1. 데이터
dataset2 = np.array(range(1,151))
size2 = 5
def new_split2 (r_data,r_size):
    new_data = []
    new_data2 = []
    for i in range(len(r_data)-(r_size-1)) : 
        new_data.append(r_data[i:i+size])
    # (146,5)    
    for i in range(len(new_data)-(r_size-1)) : 
        new_data2.append(new_data[i:i+size])
    # (142,5,5)
    return np.array(new_data2)

data3 = new_split2(dataset2,size2)
print(data3.shape)
print(type(data3))




# x_len = x.__len__()
# print("x.__len__() : ",x.__len__())
# new_x = []
# np.array(new_x)
# for i in range(x[1,].size):
#     # print("s",x[1,].size)
#     # new_x += [x[0,i],x[1,i],x[2,i]]
#     # on = np.array(x[0,i])
#     # tw = np.array(x[1,i])
#     # th = np.array(x[2,i])
#     # new_x = np.append(new_x,[on],axis=0)
#     # new_x = np.append(new_x,[tw],axis=0)
#     # new_x = np.append(new_x,[th],axis=0)
#     new_x.append([x[0,i],x[1,i],x[2,i]])
# new_x = np.array(new_x)
# print(new_x)
# print("new_x.shape >>> ",new_x.shape)










