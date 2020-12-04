import numpy as np
import pandas as pd


loan_dataset = pd.read_csv("./data/csv/train.csv",header=0,index_col=None,sep=",")
BSE_dataset = pd.read_csv("./data/csv/BSE_SENSEX_5Y.csv",header=0,index_col=0,sep=",")

# loan_DD = loan_dataset["DisbursalDate"] # 지출일
loan_DD = np.load("./data/LDD.npy",allow_pickle=True)
loan_LD = loan_dataset["loan_default"]  # loan_default

# loan_DD2 = loan_dataset["DisbursalDate"] 
# for i in range(len(loan_DD)):
#     loan_DD.iloc[i] = loan_DD.iloc[i][-2:]+loan_DD.iloc[i][3:5]+loan_DD.iloc[i][0:2]
# for i in range(len(loan_DD)):
#     loan_DD[i] = loan_DD[i][-2:]+loan_DD[i][3:5]+loan_DD[i][0:2]
dd3 = loan_dataset["loan_default"].groupby(loan_DD).mean()
# print(dd3)



bse_2018  = BSE_dataset.loc["1-August-2018":"31-October-2018"] 
print(bse_2018.loc["1-August-2018"]["Close"]) # 37521.62

# bse_d = bse_2018["Date"]
# for i in range(len(bse_d)):
#     if bse_d.iloc[i][2:5] == "Aug" or bse_d.iloc[i][2:5] == "-Au":
#         bse_d.iloc[i] = "1808"
#     if bse_d.iloc[i][2:5] == "Sep" or bse_d.iloc[i][2:5] == "-Se":
#         bse_d.iloc[i] = "1809"
#     if bse_d.iloc[i][2:5] == "Oct" or bse_d.iloc[i][2:5] == "-Oc":
#         bse_d.iloc[i] = "1810"    
# print(bse_d)

bse_c = bse_2018["Close"] 
# print(bse_c)

# date_close = []
# for i in range(len(bse_c)):
#     date_close.append([bse_d[i],  bse_c[i]])
# date_close = np.array(date_close)
# print(date_close)
# print(date_close.shape)







# 시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6)) # 단위 무엇인지 찾아볼것
plt.subplot(2,1,1)         # 2행 1열 중 첫번째
plt.plot(dd3,marker='.',c='red',label='loan_default')
plt.grid() # 모눈종이 모양으로 하겠다.

plt.title('1')
plt.ylabel('AV')
plt.xlabel('date')
plt.legend(loc='upper left')

plt.subplot(2,1,2)         # 2행 1열 중 두번째
plt.plot(bse_c,marker='.',c='blue',label="close")
plt.grid() # 모눈종이 모양으로 하겠다.

# plt.title('2')
# plt.ylabel('bse')
# plt.xlabel('date')
# plt.legend(loc='upper left')
# dd = loan_dataset.groupby("loan_default")["loan_default"].count()
# index = np.arange(len(dd))
# label = ['0', '1']
# plt.bar(index,dd)
# plt.title("LOAN DEFAULT")
# plt.xticks(index,label)
plt.show()

'''
# summarize history for accuracy
plt.plot(results.history['acc'])
plt.plot(results.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

'''