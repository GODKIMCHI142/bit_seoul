import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

datasets = load_diabetes()
x = datasets.data   # 442, 10
y = datasets.target # 442,

pca = PCA(n_components=5)
x = pca.fit_transform(x)
print(x.shape) # 442, 9

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
print(sum(pca_EVR))

