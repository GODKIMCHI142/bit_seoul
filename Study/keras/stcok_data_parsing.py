import numpy as np
import pandas as pd

samsung = pd.read_csv("./data/csv/삼성전자 1120.csv",
                        header=0,index_col=0,sep=",",encoding='CP949')

# ==================== samsung x ====================

samsung_sort   = pd.DataFrame(samsung).sort_values("일자",ascending=["True"])
samsung_x_row  = samsung_sort.loc["2018/05/04":"2020/11/17"] 
samsung_x_tv   = samsung_x_row["거래량"] 
samsung_x_ta   = samsung_x_row["금액(백만)"] 
samsung_x_o    = samsung_x_row["시가"] 
samsung_x_c    = samsung_x_row["종가"]

for i in range(len(samsung_x_tv.index)):
    samsung_x_tv.iloc[i]  = int(samsung_x_tv.iloc[i].replace(",",""))
    samsung_x_ta.iloc[i]  = int(samsung_x_ta.iloc[i].replace(",",""))
    samsung_x_o.iloc[i]   = int(samsung_x_o.iloc[i].replace(",",""))
    samsung_x_c.iloc[i]   = int(samsung_x_c.iloc[i].replace(",",""))

samsung_x_set = []

for j in range(len(samsung_x_tv.index)):
    samsung_x_set.append([int(samsung_x_ta.iloc[j]*1000000/samsung_x_tv.iloc[j]),samsung_x_o.iloc[j],samsung_x_c.iloc[j]]) 

samsung_x = np.array(samsung_x_set)

# ==================== samsung y ====================

samsung_sort2   = pd.DataFrame(samsung).sort_values("일자",ascending=["True"])
samsung_y_row   = samsung_sort2.loc["2018/05/15":"2020/11/19"]
samsung_y_c     = samsung_y_row["시가"]


for i in range(len(samsung_y_c.index)):
    samsung_y_c.iloc[i] = int(samsung_y_c.iloc[i].replace(",",""))

samsung_y_set = []
for j in range(len(samsung_y_c.index)):
     samsung_y_set.append(samsung_y_c.iloc[j])



samsung_y = np.array(samsung_y_set)


np.save("./data/samsung_x.npy", arr=samsung_x)
np.save("./data/samsung_y.npy", arr=samsung_y)
print(samsung_x.shape) # (624, 2)
print(samsung_y.shape) # (620,)

'''
print("삼성끝")

# ==================== bit computer ====================
'''
bit = pd.read_csv("./data/csv/비트컴퓨터 1120.csv",header=0,index_col=0,sep=",",encoding='CP949')                       

bit_sort   = pd.DataFrame(bit).sort_values("일자",ascending=["True"])
bit_x_row  = bit_sort.loc["2018/05/04":"2020/11/17"] 
bit_x_tv   = bit_x_row["거래량"] 
bit_x_ta   = bit_x_row["금액(백만)"] 
bit_x_o    = bit_x_row["시가"] 
bit_x_h    = bit_x_row["고가"]
bit_x_c    = bit_x_row["종가"]

for i in range(len(bit_x_tv.index)):
    bit_x_tv.iloc[i]  = int(bit_x_tv.iloc[i].replace(",",""))
    bit_x_ta.iloc[i]  = int(bit_x_ta.iloc[i].replace(",",""))
    bit_x_o.iloc[i]   = int(bit_x_o.iloc[i].replace(",",""))
    bit_x_h.iloc[i]   = int(bit_x_h.iloc[i].replace(",",""))
    bit_x_c.iloc[i]   = int(bit_x_c.iloc[i].replace(",",""))

bit_x_set = []
for j in range(len(bit_x_tv.index)):
    bit_x_set.append([int(bit_x_ta.iloc[j]*1000000/bit_x_tv.iloc[j]),
                          bit_x_o.iloc[j],
                          bit_x_h.iloc[j],
                          bit_x_c.iloc[j]]) 

bit_x = np.array(bit_x_set)
np.save("./data/bit_x.npy", arr=bit_x)

print("비트끝")
# ==================== gold ====================

gold = pd.read_csv("./data/csv/금현물.csv",header=0,index_col=0,sep=",",encoding='CP949')                       

gold_sort   = pd.DataFrame(gold).sort_values("일자",ascending=["True"])
gold_x_row  = gold_sort.loc["2018/05/04":"2020/11/17"] 
gold_x_tv   = gold_x_row["거래량"] 
gold_x_ta   = gold_x_row["거래대금(백만)"] 
gold_x_o    = gold_x_row["시가"] 
gold_x_h    = gold_x_row["고가"] 
gold_x_l    = gold_x_row["저가"] 
gold_x_c    = gold_x_row["종가"]

for i in range(len(gold_x_tv.index)):
    gold_x_tv.iloc[i]  = int(gold_x_tv.iloc[i].replace(",",""))
    gold_x_ta.iloc[i]  = int(gold_x_ta.iloc[i].replace(",",""))
    gold_x_o.iloc[i]   = int(gold_x_o.iloc[i].replace(",",""))
    gold_x_h.iloc[i]   = int(gold_x_h.iloc[i].replace(",",""))
    gold_x_l.iloc[i]   = int(gold_x_l.iloc[i].replace(",",""))
    gold_x_c.iloc[i]   = int(gold_x_c.iloc[i].replace(",",""))

gold_x_set = []

for j in range(len(gold_x_tv.index)):
    gold_x_set.append([int(gold_x_ta.iloc[j]*1000000/gold_x_tv.iloc[j]),
                           gold_x_o.iloc[j],
                           gold_x_h.iloc[j],
                           gold_x_l.iloc[j],
                           gold_x_c.iloc[j]]) 

gold_x = np.array(gold_x_set)
np.save("./data/gold_x.npy", arr=gold_x)

print("gold끝")

# ==================== kosdaq ====================

kosdaq = pd.read_csv("./data/csv/코스닥.csv",header=0,index_col=0,sep=",",encoding='CP949')                       

kosdaq_sort   = pd.DataFrame(kosdaq).sort_values("일자",ascending=["True"])
kosdaq_x_row  = kosdaq_sort.loc["2018/05/04":"2020/11/17"] 
kosdaq_x_tv   = kosdaq_x_row["거래량"] 
kosdaq_x_ta   = kosdaq_x_row["거래대금"] 
kosdaq_x_o    = kosdaq_x_row["시가"] 
kosdaq_x_h    = kosdaq_x_row["고가"] 
kosdaq_x_l    = kosdaq_x_row["저가"] 
kosdaq_x_c    = kosdaq_x_row["현재가"]
kosdaq_x_u    = kosdaq_x_row["상승"]

for i in range(len(kosdaq_x_tv.index)):
    kosdaq_x_tv.iloc[i]  = int(kosdaq_x_tv.iloc[i].replace(",",""))
    kosdaq_x_ta.iloc[i]  = int(kosdaq_x_ta.iloc[i].replace(",",""))
    kosdaq_x_o.iloc[i]   = int(kosdaq_x_o.iloc[i])
    kosdaq_x_h.iloc[i]   = int(kosdaq_x_h.iloc[i])
    kosdaq_x_l.iloc[i]   = int(kosdaq_x_l.iloc[i])
    kosdaq_x_c.iloc[i]   = int(kosdaq_x_c.iloc[i])
    kosdaq_x_u.iloc[i]   = int(kosdaq_x_u.iloc[i].replace(",",""))

kosdaq_x_set = []

for j in range(len(kosdaq_x_tv.index)):
    kosdaq_x_set.append([int(kosdaq_x_tv.iloc[j]*1000000/kosdaq_x_tv.iloc[j]*1000),
                             kosdaq_x_o.iloc[j],
                             kosdaq_x_h.iloc[j],
                             kosdaq_x_l.iloc[j],
                             kosdaq_x_u.iloc[j],
                             kosdaq_x_c.iloc[j]]) 

kosdaq_x = np.array(kosdaq_x_set)
np.save("./data/kosdaq_x.npy", arr=kosdaq_x)

print("kosdaq끝")
