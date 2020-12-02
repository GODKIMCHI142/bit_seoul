import numpy as np
import pandas as pd

from xgboost import XGBClassifier,plot_importance
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. 데이터 
x = np.load("./data/loan_x.npy")
y = np.load("./data/loan_y.npy")

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=30,shuffle=True)

model = XGBClassifier()

model.fit(x_train,y_train)

acc = model.score(x_test,y_test)

print("acc : ",acc)
# acc :  0.7822264159035834

print(model.feature_importances_)
# [0.0458775  0.02725448 0.05858004 0.02936665 0.05449096 0.
#  0.07926639 0.05172378 0.07074173 0.03006568 0.07188845 0.03431729
#  0.02645867 0.05412839 0.02906843 0.04646527 0.03964236 0.01999933
#  0.01818245 0.01690725 0.02419971 0.01932519 0.02019627 0.03452988
#  0.01664576 0.02594982 0.05472836]

ii = []
for i in range(x.shape[1]):
    if model.feature_importances_[i] == 0:
        print(i,"번째 컬럼은 0")
        ii.append(i)
    else:
        print(i," 번째 컬럼은 0아님")

print(ii) # 

if len(ii) >= 1:
    print("0 제거 for")
    for i in range(len(ii)):
        x_train = np.delete(x_train,ii[i],axis=1)
        x_test  = np.delete(x_test,ii[i],axis=1)
        print(x_train.shape)
        print(x_test.shape)

pca = PCA()
pca.fit(x_train)
pca.explained_variance_ratio_
xd = len(pca.explained_variance_ratio_)

cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum > 0.9999999999) + 1 
print(cumsum)
print(d) 

pca = PCA(n_components=d)

x_train = pca.fit_transform(x_train)
x_test = pca.fit_transform(x_test)
model = XGBClassifier()

model.fit(x_train,y_train)

acc = model.score(x_test,y_test)
print("acc : ",acc)

# acc :  0.7787952220625763

# cumsum >= 0.9999999999
# acc :  0.7734339816860029

print(model.feature_importances_)

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()
# 0 만 제거
# [0.04678704 0.03381131 0.0306039  0.03585258 0.06011136 0.02857095
#  0.03032048 0.02910494 0.10793996 0.04532629 0.06638234 0.02944729
#  0.03171631 0.03139918 0.03066294 0.02950336 0.03463577 0.05215129
#  0.03470451 0.04099752 0.03063872 0.02613236 0.02494203 0.02631598
#  0.03150592 0.03043568]


# cumsum >= 0.9999999999
# [0.09404322 0.07262214 0.06573322 0.07555147 0.1248042  0.06760624
#  0.07191535 0.06209787 0.16784981 0.07392812 0.12384835]