import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# dataset     = pd.read_csv("./data/csv/test.csv",header=0,index_col=None,sep=",")
BSE_dataset = pd.read_csv("./data/csv/BSE_SENSEX_5Y.csv",header=0,index_col=0,sep=",")
dataset     = pd.read_csv("./data/csv/train.csv",header=0,index_col=None,sep=",")
dataset.columns = dataset.columns.str.replace('.','_')

# print(dataset.shape)
# loan_default value를 count 한다.
# train.loan_default.value_counts().plot(kind='bar')
# plt.show()


# dataset에 각각 컬럼의 유니크값,갯수를 뽑아본다.
# for i in dataset.columns:
#     print(i," : column")
#     print(dataset[i].nunique()," : unique Items 갯수")
#     print(dataset[i].unique())
#     print("-"*30)
#     print("")

# 결측치 sum
# print(dataset.isna().sum()) 
# print(round(100*(dataset.isna().sum())/len(dataset), 2))
# Employment.Type : 3.29
# Employment.Type : 7661
# 결측데이터가 전체행의 몇퍼센트인지 뽑아냄




# 상관관계를 히트맵으로 표현한다.
plt.figure(figsize=(12,8)) # 12,8
sns.heatmap(dataset.corr(),annot=True, fmt=".2f") 
plt.show()


# 상관계수를 플로트 소수점2자리로 표현
# sns.heatmap(dataset.corr()) # dataset 상관관계를 히트맵으로 표현


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
#     print(unique_col[i].nunique()," : No. of unique Items") 
#     print(unique_col[i].unique())                           
#     print("-"*30)
#     print("")

# unique_col 의 i 인덱스 컬럼의 유니크값 갯수
# unique_col 의 i 인덱스 컬럼의 유니크값(들)
# 히스토그램 이용하기
# unique_col.hist(bins=20,figsize=(16,12)) # bins : 막대의 너비
# plt.show()


# UniqueID           = 모든 고객에게 제공되므로 독특하고 항상 다를 것
# MobileNo_Avl_Flag  = 사용자가 모바일 번호를 제공했는지 여부. 대출이 채무불이행 여부를 알려주지 않음
# Current_pincode_ID = 예측을 위해 필요 없는 고객 주소
# Employee_code_ID   = 사원 ID는 Loan_defualt와 관련이 없으므로 필요하지 않다.
# NO_OF_INQUIRIES    = 대출 문의 번호로는 대출이 채무불이행인지 여부를 결정하는 데 도움이 되지 않는다.
# State_ID           = 대출을 이용할 수 있는 곳이며 대출 채무불이행에 대한 예측에는 별 도움이 되지 않는다.
# branch_id          = 지점 ID가 데이터 처리와 관련이 없음
# manuator_id        = 제조업체 ID의 데이터가 너무 적음 추가하지 않음
# supplier_id        = 공급업체 ID의 데이터가 너무 적음 추가하지 않음
# UniqueID = It is provided to every customer so its Unique and will always be different
# MobileNo_Avl_Flag = Whether person provided Mobile No. Doesn't tell us if loan will default
# Current_pincode_ID = It is Customers address we don't need that for Prediction
# Employee_code_ID = Employee ID is not required as it doesn't related with Loan_defualt
# NO_OF_INQUIRIES = No. of Inquiries to loan doesn't help us to determine wheather loan will default or not
# State_ID = It is where loan is availed and doesn't add much to prediction to loan default
# branch_id = Branch ID isn't relevent to Data Processing
# manufacturer_id = Manufacturer ID doesn't add much too data
# supplier_id = Supplier ID doesn't add much too data


# 데이터를 분석 후 쓸데없는 컬럼삭제
dataset.drop(unique_col,axis=1,inplace=True) 
# dataset에 있는 unique_col 컬럼 열 삭제
# inplace = True : drop한 후의 데이터프레임으로 기존 데이터프레임을 대체하겠다
# print(dataset.head())
# print(dataset.describe(include='all').T)
# print(dataset.shape)

# print(dataset['CREDIT_HISTORY_LENGTH'].head())
# date컬럼의 값변환
def change_col_month(col):
    year = int(col.split()[0].replace('yrs',''))
    month = int(col.split()[1].replace('mon',''))
    return year*12+month

dataset['CREDIT_HISTORY_LENGTH'] = dataset['CREDIT_HISTORY_LENGTH'].apply(change_col_month)
dataset['AVERAGE_ACCT_AGE']      = dataset['AVERAGE_ACCT_AGE'].apply(change_col_month)
# dataset['CREDIT_HISTORY_LENGTH'] 모든 row를 dataset['CREDIT_HISTORY_LENGTH'] apply
# print(dataset['CREDIT_HISTORY_LENGTH'].head())


# Transform CNS Score And Create New Columns
# PERFORM_CNS_SCORE_DESCRIPTION 컬럼의 value를 카운트한다.
# print(dataset.PERFORM_CNS_SCORE_DESCRIPTION.value_counts())

# print(dataset['PERFORM_CNS_SCORE_DESCRIPTION'])
def replace_not_scored(n):
    score = n.split("-")

    if len(score) != 1:
        return score[0]
    else:
        return 'N'
dataset['CNS_SCORE_DESCRIPTION'] = dataset['PERFORM_CNS_SCORE_DESCRIPTION'].apply(replace_not_scored)

# PERFORM_CNS_SCORE_DESCRIPTION 컬럼을 replace_not_scored 넣어주고 결과값을 np.object로 만든다.
# CNS_SCORE_DESCRIPTION 라는 새로운 컬럼에 결과값을 넣어준다
# print(dataset['CNS_SCORE_DESCRIPTION'])
# print(type(dataset['CNS_SCORE_DESCRIPTION']))


# CNS_SCORE_DESCRIPTION 컬럼의 문자열을 숫자로 변환
sub_risk = {'N':-1, 'K':0, 'J':1, 'I':2, 'H':3, 'G':4, 'E':5,'F':6, 'L':7, 'M':8, 'B':9, 'D':10, 'A':11, 'C':12}
dataset['CNS_SCORE_DESCRIPTION'] = dataset['CNS_SCORE_DESCRIPTION'].apply(lambda x: sub_risk[x])

# countplot() : 범주형 데이터를 시각화 디폴트로 막대그래프임
# plt.figure(figsize=(12,8))
# sns.countplot(x=dataset['CNS_SCORE_DESCRIPTION']) 
# plt.show()


# PERFORM_CNS_SCORE_DESCRIPTION 컬럼의 값들을 replace를 통해 변환시켜준다.
dataset['PERFORM_CNS_SCORE_DESCRIPTION'].replace({'C-Very Low Risk':'Very Low Risk','A-Very Low Risk':'Very Low Risk',
                                                  'D-Very Low Risk':'Very Low Risk','B-Very Low Risk':'Very Low Risk',
                                                  'M-Very High Risk':'Very High Risk','L-Very High Risk':'Very High Risk',
                                                  'F-Low Risk':'Low Risk','E-Low Risk':'Low Risk',
                                                  'G-Low Risk':'Low Risk','H-Medium Risk':'Medium Risk',
                                                  'I-Medium Risk':'Medium Risk','J-High Risk':'High Risk',
                                                  'K-High Risk':'High Risk'},inplace=True)
                                             
# map을 만들어놓는다.                                             
risk_map = {'No Bureau History Available'                            :-1, 
            'Not Scored: No Activity seen on the customer (Inactive)':-1,
            'Not Scored: Sufficient History Not Available'           :-1,
            'Not Scored: No Updates available in last 36 months'     :-1,
            'Not Scored: Only a Guarantor'                           :-1,
            'Not Scored: More than 50 active Accounts found'         :-1,
            'Not Scored: Not Enough Info available on the customer'  :-1,
            'Very Low Risk'                                          :4,
            'Low Risk'                                               :3,
            'Medium Risk'                                            :2, 
            'High Risk'                                              :1,
            'Very High Risk'                                         :0}


dataset['PERFORM_CNS_SCORE_DESCRIPTION'] = dataset['PERFORM_CNS_SCORE_DESCRIPTION'].map(risk_map)
# PERFORM_CNS_SCORE_DESCRIPTION 컬럼의 중복되는 값들을 map함수를 통해 숫자형으로 변환
# print(dataset['PERFORM_CNS_SCORE_DESCRIPTION'].value_counts())   
                          
# sns.countplot(x=dataset['PERFORM_CNS_SCORE_DESCRIPTION']) 
# plt.show()
# countplot은 컬럼의 값을 count해준뒤 시각화한다.   
                                     

                                            
# Treating Missing Values
# print(dataset.Employment_Type.value_counts()) 

dataset['Employment_Type'] = dataset['Employment_Type'].fillna('Not_employed')
# print(dataset.Employment_Type.value_counts()) 



    # Employment_Type 컬럼의 value를 count
# Self employed    127635
# Salaried          97858  -> 합치면 전체 행수보다 적은 것으로 보아 결측치가 있음을 알 수 있다.


# fillna()를 사용하면 결측치에 값을 대입할 수 있다.
# print(dataset.Employment_Type.value_counts())
# Self employed    127635
# Salaried          97858
# Not_employed       7661

employment_map = {'Self employed':0, 'Salaried':1, 'Not_employed':-1}
dataset['Employment_Type'] = dataset['Employment_Type'].apply(lambda x : employment_map[x])
# print(dataset.Employment_Type.value_counts())

# map함수와 lambda를 동시에 사용한 모습이다.
# print(dataset.Employment_Type.value_counts())
#  0    127635
#  1     97858
# -1      7661
# sns.countplot(x=dataset['Employment_Type'])
# plt.show()


# Transforming Primary and Secondary Accounts

pri_columns = ['PRI_NO_OF_ACCTS'      ,'SEC_NO_OF_ACCTS',
               'PRI_ACTIVE_ACCTS'     ,'SEC_ACTIVE_ACCTS',
               'PRI_OVERDUE_ACCTS'    ,'SEC_OVERDUE_ACCTS',
               'PRI_CURRENT_BALANCE'  ,'SEC_CURRENT_BALANCE',
               'PRI_SANCTIONED_AMOUNT','SEC_SANCTIONED_AMOUNT',
               'PRI_DISBURSED_AMOUNT' ,'SEC_DISBURSED_AMOUNT',
               'PRIMARY_INSTAL_AMT'   , 'SEC_INSTAL_AMT']
# 첫번째 두번째 대출에 대한 컬럼들

pri_df = dataset[pri_columns]
# 그에 관한 컬럼들을 하나로 묶어 주었다.
# print(pri_df)

dataset['NO_OF_ACCTS']       = dataset['PRI_NO_OF_ACCTS']       + dataset['SEC_NO_OF_ACCTS']
dataset['ACTIVE_ACCTS']      = dataset['PRI_ACTIVE_ACCTS']      + dataset['SEC_ACTIVE_ACCTS']
dataset['OVERDUE_ACCTS']     = dataset['PRI_OVERDUE_ACCTS']     + dataset['SEC_OVERDUE_ACCTS']
dataset['CURRENT_BALANCE']   = dataset['PRI_CURRENT_BALANCE']   + dataset['SEC_CURRENT_BALANCE']
dataset['SANCTIONED_AMOUNT'] = dataset['PRI_SANCTIONED_AMOUNT'] + dataset['SEC_SANCTIONED_AMOUNT']
dataset['DISBURSED_AMOUNT']  = dataset['PRI_DISBURSED_AMOUNT']  + dataset['SEC_DISBURSED_AMOUNT']
dataset['INSTAL_AMT']        = dataset['PRIMARY_INSTAL_AMT']    + dataset['SEC_SANCTIONED_AMOUNT']
# 2개로 나누어져 있던 컬럼들을 새로운 컬럼 하나로 다시 만들었다.

dataset.drop(pri_columns, axis=1, inplace=True)
# 이후 2개로 나누어져 있던 컬럼들을 모두 삭제한다.

# print(dataset.shape)
# (233154, 26)

# new_columns = ['NO_OF_ACCTS', 'ACTIVE_ACCTS', 'OVERDUE_ACCTS', 'CURRENT_BALANCE',
#        'SANCTIONED_AMOUNT', 'DISBURSED_AMOUNT', 'INSTAL_AMT']

# for i in new_columns:
#     print(i," : distinct_value")
#     print(dataset[i].nunique()," : No. of unique Items")
#     print(dataset[i].unique())
#     print("-"*30)
#     print("")
# 새로운 컬럼의 값들과 갯수를 살펴본다. value_counts()와 비슷

# Visualization and Treating Outliers
# sns.scatterplot(data=dataset['ACTIVE_ACCTS'])
# scatterplot : 산점도 : 두 개 변수 간의 관계를 나타내는 방법이다.
# plt.show()


# 이상치 제거 -보류-
li = list(dataset['ACTIVE_ACCTS'].sort_values()[-3:].index)
dataset['ACTIVE_ACCTS'][li]  = int(dataset.drop(li)['ACTIVE_ACCTS'].mode())

li = list(dataset['NO_OF_ACCTS'].sort_values()[-4:].index)
dataset['NO_OF_ACCTS'][li]   = int(dataset.drop(li)['NO_OF_ACCTS'].mode())

li = list(dataset['OVERDUE_ACCTS'].sort_values()[-10:].index)
dataset['OVERDUE_ACCTS'][li] = int(dataset.drop(li)['OVERDUE_ACCTS'].mode())


# Lets take a look at Date of Birth Column
# 만기지급일과 생년월일로 나이 추출하기
# print(dataset.Date_of_Birth.min(),dataset.Date_of_Birth.max())
# 생년월일의 min max : 01-01-00 , 31-12-99

# print(dataset['Date_of_Birth'].head())
def age(dob):
    yr = int(dob[-2:])
    if yr >=0 and yr < 20:
        return yr + 2000
    else:
         return yr + 1900

dataset['Date_of_Birth'] = dataset['Date_of_Birth'].apply(age)
dataset['DisbursalDate'] = dataset['DisbursalDate'].apply(age)
dataset['Age']           = dataset['DisbursalDate'] - dataset['Date_of_Birth']
dataset = dataset.drop( ['DisbursalDate', 'Date_of_Birth'], axis=1)
# print(dataset['Age'].head())

# print(dataset.shape) (233154, 25)

# 나이대 시각화
# ax = plt.subplots(figsize=(10,7))
# sns.countplot(x=dataset['Age'],alpha=.8)
# plt.show()

x = dataset.drop(['loan_default'], axis=1)
y = dataset['loan_default']
# print(x.head())
# print(y.head())
# print(x.shape) # (233154, 24)
# print(y.shape) # (233154,)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1203,shuffle=True)


# select model
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, MaxAbsScaler
robustScaler = RobustScaler()
robustScaler.fit(x_train)
robustScaler.transform(x_train)
robustScaler.transform(x_test)

from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score,roc_auc_score
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

lr  = LogisticRegression()
knn = KNeighborsClassifier()
rfc = RandomForestClassifier()
dtc = DecisionTreeClassifier()
xgb = XGBClassifier()
ada = AdaBoostClassifier()
gbc = GradientBoostingClassifier()



accuracy = {}
roc_r = {}

def train_model(model,name):
    # Checking accuracy
    print("model : ",name)
    model = model.fit(x_train, y_train)
    pred  = model.predict(x_test)
    acc   = accuracy_score(y_test, pred)*100
    accuracy[model] = acc
    print('accuracy_score  : ',acc)
    print('precision_score : ',precision_score(y_test, pred)*100)
    print('recall_score    : ',recall_score(y_test, pred)*100)
    print('f1_score        : ',f1_score(y_test, pred)*100)
    roc_score = roc_auc_score(y_test, pred)*100
    roc_r[model] = roc_score
    print('roc_auc_score   : ',roc_score)
    # confusion matrix
    print('confusion_matrix')
    print(pd.DataFrame(confusion_matrix(y_test, pred)))
    fpr, tpr, threshold = roc_curve(y_test, pred)
    roc_auc = auc(fpr, tpr)*100

    print("-"*30)
    print()


# train_model(lr,"LogisticRegression")
# train_model(knn,"KNeighborsClassifier")
# train_model(rfc,"RandomForestClassifier")
# train_model(dtc,"DecisionTreeClassifier")
# train_model(xgb,"XGBClassifier")
# train_model(ada,"AdaBoostClassifier")
# train_model(gbc,"GradientBoostingClassifier")
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

parameters2 = [
    {
    "svm__learning_rate" : [0.01, 0.02, 0.03],
    "svm__n_estimators" : [200, 300, 400, 500],
    "svm__max_depth" : [6,8,10],
    "svm__colsample_bytree" : [0.6, 0.7, 0.8],
    "svm__colsample_bylevel" : [0.6, 0.7, 0.8],
    }
]
learning_rate  = [0.05]
n_estimators = [200]
max_depth = [6]
colsample_bytree = [0.6]
colsample_bylevel = [0.6]

parameters = dict(learning_rate=learning_rate, n_estimators=n_estimators, 
max_depth = [6], colsample_bylevel = [0.6], colsample_bytree = [0.6])

# parameters2 = dict(learning_rate=learning_rate2, n_estimators=n_estimators2, 
# max_depth =max_depth2, colsample_bylevel = colsample_bytree2, colsample_bytree = colsample_bylevel2)


model = RandomizedSearchCV(XGBClassifier(),parameters,cv=3,verbose=2)
model.fit(x_train,y_train)
acc = model.score(x_test,y_test)
pred  = model.predict(x_test)
acc   = accuracy_score(y_test, pred)*100
accuracy[model] = acc
print('accuracy_score  : ',acc)
print('precision_score : ',precision_score(y_test, pred)*100)
print('recall_score    : ',recall_score(y_test, pred)*100)
print('f1_score        : ',f1_score(y_test, pred)*100)
roc_score = roc_auc_score(y_test, pred)*100
roc_r[model] = roc_score
print('roc_auc_score   : ',roc_score)
# confusion matrix
print('confusion_matrix')
print(pd.DataFrame(confusion_matrix(y_test, pred)))

model = model.best_estimator_


n     = 0
b_acc = acc

thresholds = np.sort(model.feature_importances_)
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    

    select_x_train = selection.transform(x_train)
    selection_model = XGBClassifier()
    selection_model.fit(select_x_train,y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    acc = selection_model.score(select_x_test,y_test)
    acc_score = accuracy_score(y_test,y_predict)
    if acc > b_acc:
        n = select_x_train.shape[1]
        b_acc = acc
        L_selection = selection
        print("Thresh=%.3f, n=%d, acc: %.15f%%, acc_score: %.15f%%"%(thresh,select_x_train.shape[1],acc,acc_score))
    

x_train = L_selection.transform(x_train)
x_test = L_selection.transform(x_test)
# x_predict = L_selection.transform(x_predict)

pipe = Pipeline([("scaler",MinMaxScaler()), ("svm",XGBClassifier())])
model = RandomizedSearchCV(pipe,parameters2,cv=3,verbose=2)
model.fit(x_train,y_train)

acc = model.score(x_test,y_test)
pred  = model.predict(x_test)
acc   = accuracy_score(y_test, pred)*100
accuracy[model] = acc
print('accuracy_score  : ',acc)
print('precision_score : ',precision_score(y_test, pred)*100)
print('recall_score    : ',recall_score(y_test, pred)*100)
print('f1_score        : ',f1_score(y_test, pred)*100)
roc_score = roc_auc_score(y_test, pred)*100
roc_r[model] = roc_score
print('roc_auc_score   : ',roc_score)
# confusion matrix
print('confusion_matrix')
print(pd.DataFrame(confusion_matrix(y_test, pred)))

# accuracy_score  :  78.17204454801492
# precision_score :  54.6875
# recall_score    :  0.6869030485411488
# f1_score        :  1.3567644398501097
# roc_auc_score   :  50.263870104245335
# confusion_matrix
#        0    1
# 0  54574   87
# 1  15181  105



'''
model :  LogisticRegression
accuracy_score  :  78.22478608650898
precision_score :  0.0
recall_score    :  0.0
f1_score        :  0.0
roc_auc_score   :  50.0
confusion_matrix
       0  1
0  36477  0
1  10154  0
------------------------------

model :  KNeighborsClassifier
accuracy_score  :  74.27247968089897
precision_score :  26.96825793551612
recall_score    :  10.626354146149302
f1_score        :  15.24549629106323
roc_auc_score   :  51.307913482318824
confusion_matrix
       0     1
0  33555  2922
1   9075  1079
------------------------------

model :  RandomForestClassifier
accuracy_score  :  77.11393708048294
precision_score :  33.02752293577982
recall_score    :  4.9635611581642705
f1_score        :  8.63013698630137
roc_auc_score   :  51.080897831049136
confusion_matrix
       0     1
0  35455  1022
1   9650   504
------------------------------

model :  DecisionTreeClassifier
accuracy_score  :  66.49868113486737
precision_score :  24.954195676071823
recall_score    :  26.826866259602127
f1_score        :  25.856668248694824
roc_auc_score   :  52.18443951738777
confusion_matrix
       0     1
0  28285  8192
1   7430  2724
------------------------------

model :  XGBClassifier
accuracy_score  :  78.16474019429135
precision_score :  46.27659574468085
recall_score    :  1.7136103998424266
f1_score        :  3.3048433048433052
roc_auc_score   :  50.579918394537
confusion_matrix
       0    1
0  36275  202
1   9980  174
------------------------------

model :  AdaBoostClassifier
accuracy_score  :  78.21835259805708
precision_score :  49.01960784313725
recall_score    :  0.7386251723458735
f1_score        :  1.4553216260793636
roc_auc_score   :  50.26239589894536
confusion_matrix
       0   1
0  36399  78
1  10079  75
------------------------------

model :  GradientBoostingClassifier
accuracy_score  :  78.22907507881023
precision_score :  51.724137931034484
recall_score    :  0.29545006893834946
f1_score        :  0.5875440658049353
roc_auc_score   :  50.109344685207994
confusion_matrix
       0   1
0  36449  28
1  10124  30


'''










