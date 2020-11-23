import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score

# 1. data
dataset = pd.read_csv("./data/csv/winequality-white.csv",header=0,index_col=None,sep=";")

count_data = dataset.groupby("quality")["quality"].count()
# 퀄리티 컬럼에 있는 개체들을 카운트 하겠다
print(count_data)
# quality
# 3      20
# 4     163
# 5    1457
# 6    2198
# 7     880
# 8     175
# 9       5

import matplotlib.pyplot as plt
count_data.plot()
plt.show()