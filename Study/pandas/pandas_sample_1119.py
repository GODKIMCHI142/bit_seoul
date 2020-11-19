import pandas as pd
import numpy as np

from numpy.random import randn
np.random.seed(100)


data = randn(5,4) # 5행 4열
print(data)
df = pd.DataFrame(data, index='A B C D E'.split(), columns='가 나 다 라'.split())
print(df)


data2 = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]] # 5,4
df2 = pd.DataFrame(data2, index=['A','B','C','D','E'], columns=['가','나','다','라'])
print(df2)
#    가   나   다   라
# A   1   2   3   4
# B   5   6   7   8
# C   9  10  11  12
# D  13  14  15  16
# E  17  18  19  20

dff = pd.DataFrame(data2)
print(dff)
#     0   1   2   3
# 0   1   2   3   4
# 1   5   6   7   8
# 2   9  10  11  12
# 3  13  14  15  16
# 4  17  18  19  20

print("dff[0] : \n",dff[0])
# 0     1
# 1     5
# 2     9
# 3    13
# 4    17



df3 = pd.DataFrame(np.array([[1,2,3],[1,2,3]]))
print(df3)

# 컬럼
print("df2['나'] : \n",df2['나']) 
# A     2
# B     6
# C    10
# D    14
# E    18
# Name: 나, dtype: int64

print("df2[['나','라']] : \n",df2[['나','라']]) 
#    나   라
# A   2   4
# B   6   8
# C  10  12
# D  14  16
# E  18  20

# print("df2[0] : \n",df2[0]) # error 컬럼명으로 해주어야함


# ROW

# print("df2.loc['나'] : \n",df2.loc['나']) # error, location은 행일때 사용한다. 

print("df2.iloc['A'] : \n",df2.loc['A'])  # loc  : location
# 가    1
# 나    2
# 다    3
# 라    4

print("df2.iloc[:,1] : \n",df2.iloc[:,1]) # iloc : index location
# A     2
# B     6
# C    10
# D    14
# E    18

print("df2.iloc[['A','C']] : \n",df2.loc[['A','C']])  # loc  : location
#    가   나   다   라
# A  1   2   3   4
# C  9  10  11  12

# print("df2.iloc[0] : \n",df2.loc[0])  # error loc
print("df2.iloc[[0,2]] : \n",df2.iloc[[0,2]]) # iloc : index location
#    가   나   다   라
# A  1   2   3   4
# C  9  10  11  12

# 행렬
print("df2.loc[['A','B'],['나']['다']] : \n",df2.loc[['A','B'],['나','다']])
#    나  다
# A  2  3
# B  6  7

# 한개의 값만 확인
print("df2.loc['E','다'] : \n",df2.loc['E','다']) # 19
print("df2.iloc[4,2] : \n",df2.iloc[4,2])         # 19
print("df2.iloc[4][2] : \n",df2.iloc[4][2])       # 19






