import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# dataset     = pd.read_csv("./data/csv/test.csv",header=0,index_col=None,sep=",")
BSE_dataset = pd.read_csv("./data/csv/BSE_SENSEX_5Y.csv",header=0,index_col=0,sep=",")
dataset     = pd.read_csv("./data/csv/train.csv",header=0,index_col=None,sep=",")
dataset.columns = dataset.columns.str.replace('.','_')

# loan_default value를 count
# train.loan_default.value_counts().plot(kind='bar')
# plt.show()

# train.loan_default.value_counts()


# 상관관계를 히트맵으로 표현한다.
# plt.figure(figsize=(12,8)) # 12,8
# sns.heatmap(dataset.corr(),annot=True, fmt=".2f") # 상관계수를 플로트 소수점2자리로 표현
# # sns.heatmap(dataset.corr()) # dataset 상관관계를 히트맵으로 표현
# plt.show()

# dataset에 loan_default를 기준으로하여 상관관계 밸류를 내림차순으로 정렬해본다.
# print(dataset.corr()['loan_default'].sort_values(ascending = False))
# loan_default                           1.000000
# ltv                                    0.098208
# disbursed_amount                       0.077675
# State_ID                               0.048075
# VoterID_flag                           0.043747
# NO.OF_INQUIRIES                        0.043678
# PRI.OVERDUE.ACCTS                      0.040872
# DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS    0.034462
# UniqueID                               0.033848
# branch_id                              0.030193
# Current_pincode_ID                     0.028419
# supplier_id                            0.027357
# Employee_code_ID                       0.020657
# asset_cost                             0.014261
# PAN_flag                               0.002046
# SEC.OVERDUE.ACCTS                     -0.001371
# SEC.INSTAL.AMT                        -0.001548
# SEC.CURRENT.BALANCE                   -0.005531
# Driving_flag                          -0.005821
# SEC.ACTIVE.ACCTS                      -0.005993
# SEC.DISBURSED.AMOUNT                  -0.006248
# SEC.SANCTIONED.AMOUNT                 -0.006354
# Passport_flag                         -0.007602
# SEC.NO.OF.ACCTS                       -0.008385
# PRIMARY.INSTAL.AMT                    -0.010616
# PRI.DISBURSED.AMOUNT                  -0.011155
# PRI.SANCTIONED.AMOUNT                 -0.011304
# manufacturer_id                       -0.025039
# PRI.CURRENT.BALANCE                   -0.027386
# NEW.ACCTS.IN.LAST.SIX.MONTHS          -0.029400
# PRI.NO.OF.ACCTS                       -0.035456
# PRI.ACTIVE.ACCTS                      -0.041451
# Aadhar_flag                           -0.041593
# PERFORM_CNS.SCORE                     -0.057929
# MobileNo_Avl_Flag                           NaN
# Name: loan_default, dtype: float64


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


#Lets Look at few columns
columns_unique = ['UniqueID','MobileNo_Avl_Flag','Current_pincode_ID','Employee_code_ID',
                  'NO_OF_INQUIRIES','State_ID','branch_id','manufacturer_id','supplier_id']

unique_col = dataset[columns_unique]
# print(unique_col.head())

#Looking at all unique_col values
# for i in unique_col.columns:
#     print(i," : distinct_value")
#     print(unique_col[i].nunique()," : No. of unique Items") # unique_col 의 i 인덱스 컬럼의 유니크값 갯수
#     print(unique_col[i].unique())                           # unique_col 의 i 인덱스 컬럼의 유니크값(들)
#     print("-"*30)
#     print("")

# 히스토그램 이용하기
# unique_col.hist(bins=20,figsize=(16,12)) # bins : 막대의 너비
# plt.show()

# UniqueID = It is provided to every customer so its Unique and will always be different
# MobileNo_Avl_Flag = Whether person provided Mobile No. Doesn't tell us if loan will default
# Current_pincode_ID = It is Customers address we don't need that for Prediction
# Employee_code_ID = Employee ID is not required as it doesn't related with Loan_defualt
# NO_OF_INQUIRIES = No. of Inquiries to loan doesn't help us to determine wheather loan will default or not
# State_ID = It is where loan is availed and doesn't add much to prediction to loan default
# branch_id = Branch ID isn't relevent to Data Processing
# manufacturer_id = Manufacturer ID doesn't add much too data
# supplier_id = Supplier ID doesn't add much too data














