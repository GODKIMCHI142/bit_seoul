import numpy as np
import pandas as pd

datasets = pd.read_csv("./data/csv/iris_ys.csv",
                        header=0,index_col=2,sep=",")

print(datasets)
print(datasets.shape)


# header, index_col에 따른 데이터 변화

# header = None, index_col = None
'''
         0             1            2             3            4        5
0      NaN  sepal_length  sepal_width  petal_length  petal_width  species
1      1.0           5.1          3.5           1.4          0.2        0
2      2.0           4.9            3           1.4          0.2        0
149  149.0           6.2          3.4           5.4          2.3        2
150  150.0           5.9            3           5.1          1.8        2

[151 rows x 6 columns]
(151, 6)
'''
# header = None, index_col = 0
'''
                  1            2             3            4        5
0
NaN    sepal_length  sepal_width  petal_length  petal_width  species
1.0             5.1          3.5           1.4          0.2        0
2.0             4.9            3           1.4          0.2        0
149.0           6.2          3.4           5.4          2.3        2
150.0           5.9            3           5.1          1.8        2

[151 rows x 5 columns]
(151, 5)
'''
# header = None, index_col = 1
'''
                  0            2             3            4        5
1
sepal_length    NaN  sepal_width  petal_length  petal_width  species
5.1             1.0          3.5           1.4          0.2        0
4.9             2.0            3           1.4          0.2        0
6.2           149.0          3.4           5.4          2.3        2
5.9           150.0            3           5.1          1.8        2

[151 rows x 5 columns]
(151, 5)
'''
# header = 0, index_col = None
'''
     Unnamed: 0  sepal_length  sepal_width  petal_length  petal_width  species
0             1           5.1          3.5           1.4          0.2        0
1             2           4.9          3.0           1.4          0.2        0
148         149           6.2          3.4           5.4          2.3        2
149         150           5.9          3.0           5.1          1.8        2

[150 rows x 6 columns]
(150, 6)
'''
# header = 0, index_col = 0
'''
     sepal_length  sepal_width  petal_length  petal_width  species
1             5.1          3.5           1.4          0.2        0
2             4.9          3.0           1.4          0.2        0
149           6.2          3.4           5.4          2.3        2
150           5.9          3.0           5.1          1.8        2

[150 rows x 5 columns]
(150, 5)
'''
# header = 0, index_col = 1
'''
              Unnamed: 0  sepal_width  petal_length  petal_width  species
sepal_length
5.1                    1          3.5           1.4          0.2        0
4.9                    2          3.0           1.4          0.2        0
6.2                  149          3.4           5.4          2.3        2
5.9                  150          3.0           5.1          1.8        2

[150 rows x 5 columns]
(150, 5)
'''
# header = 1, index_col = None
'''
       1  5.1  3.5  1.4  0.2  0
0      2  4.9  3.0  1.4  0.2  0
1      3  4.7  3.2  1.3  0.2  0
147  149  6.2  3.4  5.4  2.3  2
148  150  5.9  3.0  5.1  1.8  2

[149 rows x 6 columns]
(149, 6)
'''
# header = 1, index_col = 0
'''
     5.1  3.5  1.4  0.2  0
1
2    4.9  3.0  1.4  0.2  0
149  6.2  3.4  5.4  2.3  2
150  5.9  3.0  5.1  1.8  2

[149 rows x 5 columns]
(149, 5)
'''
# header = 1, index_col = 1
'''
       1  3.5  1.4  0.2  0
5.1
4.9    2  3.0  1.4  0.2  0
6.2  149  3.4  5.4  2.3  2
5.9  150  3.0  5.1  1.8  2

[149 rows x 5 columns]
(149, 5)
'''

# print(datasets.head())
# print(datasets.tail())


# aaa = datasets.values
# aaa = datasets.to_numpy()
# print(type(datasets))
# print(type(aaa))
# print(datasets.shape)
# print(aaa.shape)

# np.save("./data/iris_ys_pd.npy",arr=aaa)






