import numpy as np
import pandas as pd

datasets = pd.read_csv("./data/csv/삼성전자 1120.csv",
                        header=0,index_col=0,sep=",",encoding='CP949')

print(datasets)
print(datasets.shape)

dataset_1 = pd.DataFrame(datasets).sort_values("일자",ascending=["True"])


dataset_2 = dataset_1.loc["2018/05/04":"2020/11/09"] # 627 , 16

dataset_3 = dataset_2["거래량"] # 627 , 2

dataset_33 = dataset_2["금액(백만)"] # 627 , 2
print(dataset_33)
dataset_4 = dataset_2["종가"]

print(dataset_4)





for i in range(618):
    dataset_4.iloc[i] = int(dataset_4.iloc[i].replace(",",""))

for i in range(len(dataset_3.index)):
    dataset_3.iloc[i] = int(dataset_3.iloc[i].replace(",",""))

for i in range(len(dataset_33.index)):
    dataset_33.iloc[i] = int(dataset_33.iloc[i].replace(",",""))

new_dataset = []
new_dataset2 = []
for i in range(1) :
    for j in range(618):
         new_dataset.append(int(dataset_33.iloc[j]*1000000/dataset_3.iloc[j])) 
# print(new_dataset)


for j in range(618):
     new_dataset2.append(dataset_4.iloc[j])

# print(new_dataset2)

print(new_dataset[0])

new_dataset3 = []
for i in range(1):
    for j in range(618):
         new_dataset3.append([new_dataset[j],new_dataset2[j]]) 

print(new_dataset3)

new_x = np.array(new_dataset3)
print(new_x.shape)

dataset_1 = pd.DataFrame(datasets).sort_values("일자",ascending=["True"])
dataset_2 = dataset_1.loc["2018/05/14":"2020/11/10"] # 618 , 16
dataset_4 = dataset_2["종가"]
print(dataset_4)

print(dataset_4)
for i in range(614):
    dataset_4.iloc[i] = int(dataset_4.iloc[i].replace(",",""))

new_dataset22 = []
for j in range(614):
     new_dataset22.append(dataset_4.iloc[j])

print(new_dataset22)

new_y = np.array(new_dataset22)
print(new_y.shape)

np.save("./data/samsung_x_1.npy", arr=new_x)
np.save("./data/samsung_y_1.npy", arr=new_y)


# print(dataset_4)
# datasets = pd.read_csv("./data/csv/iris_ys.csv",
#                         header=0,index_col=2,sep=",")

# print(datasets)
# print(datasets.shape)

