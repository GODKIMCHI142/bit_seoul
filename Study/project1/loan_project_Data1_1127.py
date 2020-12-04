import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# dataset     = pd.read_csv("./data/csv/test.csv",header=0,index_col=None,sep=",")
dataset     = pd.read_csv("./data/csv/train.csv",header=0,index_col=None,sep=",")
# dataset.columns=dataset.columns.str.replace('.','_')
BSE_dataset = pd.read_csv("./data/csv/BSE_SENSEX_5Y.csv",header=0,index_col=0,sep=",")

# 상관관계를 히트맵으로 표현한다.
# plt.figure(figsize=(12,8)) # 12,8
# sns.heatmap(dataset.corr(),annot=True, fmt=".2f")
# # sns.heatmap(dataset.corr())
# plt.show()


# dataset에 컬럼명을 뽑아본다.
# columns = dataset.columns
# print(columns)
# Index(['UniqueID', 'disbursed_amount', 'asset_cost', 'ltv', 'branch_id',
#        'supplier_id', 'manufacturer_id', 'Current_pincode_ID', 'Date.of.Birth',
#        'Employment.Type', 'DisbursalDate', 'State_ID', 'Employee_code_ID',
#        'MobileNo_Avl_Flag', 'Aadhar_flag', 'PAN_flag', 'VoterID_flag',
#        'Driving_flag', 'Passport_flag', 'PERFORM_CNS.SCORE',
#        'PERFORM_CNS.SCORE.DESCRIPTION', 'PRI.NO.OF.ACCTS', 'PRI.ACTIVE.ACCTS',
#        'PRI.OVERDUE.ACCTS', 'PRI.CURRENT.BALANCE', 'PRI.SANCTIONED.AMOUNT',
#        'PRI.DISBURSED.AMOUNT', 'SEC.NO.OF.ACCTS', 'SEC.ACTIVE.ACCTS',
#        'SEC.OVERDUE.ACCTS', 'SEC.CURRENT.BALANCE', 'SEC.SANCTIONED.AMOUNT',
#        'SEC.DISBURSED.AMOUNT', 'PRIMARY.INSTAL.AMT', 'SEC.INSTAL.AMT',
#        'NEW.ACCTS.IN.LAST.SIX.MONTHS', 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',
#        'AVERAGE.ACCT.AGE', 'CREDIT.HISTORY.LENGTH', 'NO.OF_INQUIRIES',
#        'loan_default'],
#       dtype='object')





# LOAN X
# loan_DA     = dataset["disbursed_amount"]      # 대출금총액
# loan_AC     = dataset["asset_cost"]            # 재산액
# loan_LTV    = dataset["ltv"]                   # 재산액 대비 대출
# loan_DOB    = dataset["Date.of.Birth"]         # 생년
# loan_ET     = dataset["Employment.Type"]       # 직업유형
# loan_MAF    = dataset["MobileNo_Avl_Flag"]     # 휴대폰번호를 공유하였다면 1
# loan_AF     = dataset["Aadhar_flag"]           # 주민등록번호를 공유하였다면 1
# loan_PF     = dataset["PAN_flag"]              # 소득세 번호를 공유하였다면 1
# loan_VF     = dataset["VoterID_flag"]          # 유권자ID를 공유하였다면 1
# loan_PAF    = dataset["Passport_flag"]         # 여권을 공유하였다면 1
# loan_PCS    = dataset["PERFORM_CNS.SCORE"]     # 신용 점수
# loan_PNOC   = dataset["PRI.NO.OF.ACCTS"]       # 대출금 지불 당시 대출받은 총 횟수
# loan_PAA    = dataset["PRI.ACTIVE.ACCTS"]      # 대출금 지불 당시 활성대출 수
# loan_POA    = dataset["PRI.OVERDUE.ACCTS"]     # 대출금 지불 당시 채무불이행 계좌수 
# loan_PCB    = dataset["PRI.CURRENT.BALANCE"]   # 대출금 지불 당시 활성대출의 미지급액
# loan_PSA    = dataset["PRI.SANCTIONED.AMOUNT"] # 대출금 지불 당시 모든대출에 대해 허가된 총액
# loan_PDA    = dataset["PRI.DISBURSED.AMOUNT"]  # 대출금 지불 당시 모든대출에 대해 지출된 총액
# loan_SNOA   = dataset["SEC.NO.OF.ACCTS"]       # 대출금 지불 당시 받은 총 대출의 수
# loan_SAA    = dataset["SEC.ACTIVE.ACCTS"]      # 대출금 지불 당시 활성대출 수
# loan_SOA    = dataset["SEC.OVERDUE.ACCTS"]     # 대출금 지불 당시 채무불이행 계좌수
# loan_SCB    = dataset["SEC.CURRENT.BALANCE"]   # 대출금 지불 당시 활성대출의 미지급액
# loan_SSA    = dataset["SEC.SANCTIONED.AMOUNT"] # 대출금 지불 당시 모든대출에 대해 허가된 총액
# loan_SDA    = dataset["SEC.DISBURSED.AMOUNT"]  # 대출금 지불 당시 모든대출에 대해 지출된 총액
# loan_PIA    = dataset["PRIMARY.INSTAL.AMT"]    # 첫번째 대출에 대한 EMI 총금액
# loan_SIA    = dataset["SEC.INSTAL.AMT"]        # 두번째 대출에 대한 EMI 총금액
# loan_NAILSM = dataset["NEW.ACCTS.IN.LAST.SIX.MONTHS"]         # 대출금 지불 전 6개월 동안 고객이 새로 받은 대출
# loan_DAILSM = dataset["DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS"]  # 지난 6개월 동안 채무불이행한 대출 수



# for i in range(len(loan_DOB)):
#     loan_DOB.iloc[i] = loan_DOB.iloc[i][-2:]
    
# for i in range(len(loan_DOB)):
#     x = loan_ET.iloc[i]
#     if x == "Salaried":
#         loan_ET.iloc[i] = 0
#     elif x == "Self employed":
#         loan_ET.iloc[i] = 1
#     else:
#         loan_ET.iloc[i] = 2
# # print(loan_ET[:100])

loan_DD = dataset["DisbursalDate"] 
for i in range(len(loan_DD)):
    loan_DD.iloc[i] = int(loan_DD.iloc[i][-2:]+loan_DD.iloc[i][3:5]+loan_DD.iloc[i][0:2])
    
print(loan_DD.sort_values())

bse_2018  = BSE_dataset.loc["1-August-2018":"31-October-2018"] 
# BSE = ""
loan_DOB= []
# loan_x = []

for i in range(len(loan_DOB)):
    if loan_DD[i][2:4] == "08":
        BSE = bse_2018.loc["31-August-2018"]["Close"]
    elif loan_DD[i][2:4] == "09":
        BSE = bse_2018.loc["28-September-2018"]["Close"]
    else :
        BSE = bse_2018.loc["31-October-2018"]["Close"]
        
#     loan_x.append([
#         loan_DA[i],  loan_AC[i],   loan_LTV[i],  loan_DOB[i], loan_ET[i],
#         loan_MAF[i], loan_AF[i],   loan_PF[i],   loan_VF[i],  loan_PAF[i],
#         loan_PCS[i], loan_PNOC[i], loan_PAA[i],  loan_POA[i], loan_PCB[i],
#         loan_PSA[i], loan_PDA[i],  loan_SNOA[i], loan_SAA[i], loan_SOA[i],
#         loan_SCB[i], loan_SSA[i],  loan_SDA[i],  loan_PIA[i], loan_SIA[i], 
#         loan_NAILSM[i], loan_DAILSM[i], BSE
#     ])


# loan_x = np.array(loan_x)
# print(loan_x)
# print(loan_x.shape)
# print("xend")

# np.save("./data/loan_x_predict.npy", arr=loan_x)

'''
# ========================================================================================

# loan_y
loan_LD = dataset["loan_default"] # 만기일 첫번째 EMI 지불 채무불이행

loan_y = []
for j in range(len(loan_LD)):
     loan_y.append(loan_LD.iloc[j])
loan_y = np.array(loan_y)
print(loan_y[:10])
print("yend")


np.save("./data/loan_x.npy", arr=loan_x)
np.save("./data/loan_y.npy", arr=loan_y)
print(loan_x.shape) # 
print(loan_y.shape) # 


'''


# xx = dataset["DisbursalDate"] 
# c7 = 0
# c8 = 0
# c9 = 0
# for i in range(len(xx)):
    
#     xx.iloc[i] = int(xx.iloc[i][-2:])
#     xxx = xx.iloc[i]
#     if xxx == 18:
#         c8 = c8+1
#         print(c8)
#     elif xxx == 19:
#         c9 = c9+1
#     elif xxx == 17:
#         c7 = c7+1
#     else:
#         print(xxx)
# print(c7)
# print(c8)
# print(c9)

# count_data = dataset.groupby("DisbursalDate")["DisbursalDate"].count()
# print(count_data)
# xx = dataset["DisbursalDate"] 
# for i in range(len(xx)):
#     xx.iloc[i] = xx.iloc[i][-2:]+xx.iloc[i][3:5]+xx.iloc[i][0:2]
# xx = np.array(xx)
# np.save("./data/LDD.npy", arr=xx)

# dd1 = dataset["loan_default"].groupby(xx).size()
# print(dd1)
# dd2 = dataset["loan_default"].groupby(xx).sum()
# print(dd2)
# dd3 = dataset["loan_default"].groupby(xx).mean()
# print(dd3)
# dd4 = dataset["loan_default"].groupby(dataset["loan_default"]).count()
# print(dd4)

# 0    182543
# 1     50611

# DisbursalDate
# 01-08-18    1708
# 02-10-18      25
# 03-08-18    1666
# 03-09-18    1406
# 03-10-18    2076
#             ...
# 30-08-18    4664
# 30-09-18    3601
# 30-10-18    5837
# 31-08-18    6690
# 31-10-18    8826
# Name: DisbursalDate, Length: 84, dtype: int64