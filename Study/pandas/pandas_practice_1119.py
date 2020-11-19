#    가   나   다   라
# A   1   2   3   4
# B   5   6   7   8
# C   9  10  11  12
# D  13  14  15  16
# E  17  18  19  20

import numpy as np
import pandas as pd

data = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]]) # 5행 4열
df = pd.DataFrame(data,index="A B C D E".split(), columns="가 나 다 라".split())
print(df)

# 가 다
# print(df[['가','다']])

# 나
# print(df[['나']])

# 라
# print(df[['라']])


# B C D
# print(df.loc[['B','C','D']])
# print(df.iloc[[1,2,3]])

# A D E
# print(df.loc[['A','D','E']])
# print(df.iloc[[0,3,4]])

# A B C D E
# print(df.loc[:])
# print(df.iloc[:])


# 가 나 다 라 / B E
# print(df.loc[['B','E'],['가','나','다','라']])
# print(df.iloc[[1,4],[0,1,2,3]])


# 가 라 / A D E
# print(df.loc[['A','D','E'],['가','라']])
# print(df.iloc[[0,3,4],[0,3]])

# 나 다 / C D E
# print(df.loc[['C','D','E'],['나','다']])
# print(df.iloc[[2,3,4],[1,2]])

# 가 B
# print(df.loc['B']['가']) # 5
# print(df.loc[['B'],['가']]) # 5
# print(df.iloc[1][0]) # 5
# print(df.iloc[1,0]) # 5 [,] 행렬
# print(df.iloc[[0,1]]) # [[,]] 행

# 나 E
# print(df.loc['E']['나']) # 18
# print(df.loc[['E'],['나']]) # 18
# print(df.iloc[4][1]) # 18
# print(df.iloc[4,1]) # 18 [] 행렬


# 라 A