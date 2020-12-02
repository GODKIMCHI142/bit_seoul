import numpy as np
from sklearn.decomposition import PCA

x = np.load("./data/loan_x.npy")
y = np.load("./data/loan_y.npy")

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
x_PCAEVR = pca.explained_variance_ratio_
print(x_PCAEVR)


print(cumsum)


d = np.argmax(cumsum >= 0.9999999999) + 1 
print(d) 



import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()
