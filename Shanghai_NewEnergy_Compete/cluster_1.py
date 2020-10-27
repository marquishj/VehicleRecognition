# -*- coding: utf-8 -*-
# @Time    : 2020/10/27 13:30
# @Author  : Hou Jue
# @Email   : 382062104@qq.com
# @File    : cluster_1.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)
from sklearn.cluster import KMeans,MiniBatchKMeans
#训练模型
k=4
kmeans = KMeans(n_clusters=k)
minimeans=MiniBatchKMeans(n_clusters=k)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
y_kmeans2 = minimeans.fit_predict(X)
#绘制图像
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.show()
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans2, s=50, cmap='viridis')
centers2 = minimeans.cluster_centers_
plt.scatter(centers2[:, 0], centers2[:, 1], c='black', s=200, alpha=0.5);
plt.show()