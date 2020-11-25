import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

datasets = load_diabetes()
x = datasets.data   # 442, 10
y = datasets.target # 442,

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
# cumsum : 누적된 합
xx = pca.explained_variance_ratio_
print(xx)
# [0.40242142 0.14923182 0.12059623 0.09554764 0.06621856 0.06027192
#  0.05365605 0.04336832 0.00783199 0.00085605]

print(cumsum)
# [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759
#  0.94794364 0.99131196 0.99914395 1.        ]

d = np.argmax(cumsum >= 0.95) + 1 
print(cumsum >= 0.95)
# [False False False False False False False  True  True  True]
print(d) 
# 8


import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()
