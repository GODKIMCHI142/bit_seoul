import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA

(x_train, _), (x_test, _) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
print(x.shape) # (70000, 28, 28)

# 실습
# pca를 통해 0.95일 때 컬럼수는 몇개?
# 그래프 그리기

x = x.reshape(x.shape[0],(x.shape[1]*x.shape[2]))

'''
# n_components
pca = PCA(n_components=700)
x2d = pca.fit_transform(x)
print(x2d.shape) # 70000, 700

pca_EVR = pca.explained_variance_ratio_
# 중요도가 가장 큰 순서부터 나온다.
print(pca_EVR)
print(sum(pca_EVR))


'''
# cumsum
pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
# cumsum : 누적된 합

print(cumsum)
print(x.shape)

d = np.argmax(cumsum >= 0.95) + 1 # index라서 +1한듯
print(cumsum >= 0.95)
print(d) # 87
# f = np.argmax(cumsum) + 1# index라서 +1한듯
# print(f) # 71


import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()

