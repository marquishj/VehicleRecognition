# -*- coding: utf-8 -*-
# @Time    : 2020/10/27 11:13
# @Author  : Hou Jue
# @Email   : 382062104@qq.com
# @File    : cluster.py
# @Software: PyCharm


class KMeans():
    def __init__(self, n_clusters: int = 8, max_iter: int = 300, tol: float = 1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.c_clusters = None

    def cluster(self, lo_c, lo_x):
        '''
        根据聚类中心确定每个点得类别
        lo_c:聚类中心位置
        lo_x:数据
        '''
        ## 得到每个点到聚类中心得距离
        op = []
        for i in range(len(lo_c)):
            op.append(np.linalg.norm(lo_c[i, :] - lo_x, axis=1))
        ## 根据距离最小确定每个点得类别
        return np.argmin(op, axis=0)

    def updateCenter(self, lo_x, label):
        lo_c = []
        for n in range(self.n_clusters):
            lo_c.append(np.mean(lo_x[label == n], axis=0))
        return lo_c

    def score(self, lo_cOld, lo_cNew):
        return np.sum(np.linalg.norm(lo_cOld - lo_cNew, axis=1))

    def plotFigure(self, lo_c, X, label, i):
        ax = plt.subplot(4, 4, i)
        ax.scatter(X[:, 0], X[:, 1], c=label, cmap=plt.cm.Paired)
        ax.scatter(lo_c[:, 0], lo_c[:, 1], c='g', marker='o', s=300, alpha=0.5)

    def fit(self, X):
        index = np.random.randint(0, len(X), self.n_clusters)
        self.c_clusters = X[index, :]
        lo_cOld = np.copy(self.c_clusters) - 100

        label = self.cluster(self.c_clusters, X)
        Score = self.score(lo_cOld, self.c_clusters)
        inter = 1

        plt.figure(figsize=(4 * 2, 4 * 3))
        self.plotFigure(self.c_clusters, X, label, inter)
        while Score > self.tol and inter < self.max_iter:
            print('--------------Score', Score)
            lo_cOld = np.copy(self.c_clusters)
            # print('--------------第%d次聚类中心%s'%(inter,self.c_clusters ))
            self.c_clusters = np.array(self.updateCenter(X, label))
            # print('--------------第%d次聚类中心%s'%(inter,self.c_clusters ))
            label = self.cluster(self.c_clusters, X)
            Score = self.score(lo_cOld, self.c_clusters)
            inter += 1
            if inter % 1 == 0 and inter < 300:
                self.plotFigure(self.c_clusters, X, label, inter)

    def predict(self, X_test):
        label_ = self.cluster(self.c_clusters, X_test)
        plt.figure(figsize=(4, 4))
        plt.scatter(X_test[:, 0], X_test[:, 1], c=label_, cmap='cool')
        plt.scatter(self.c_clusters[:, 0], self.c_clusters[:, 1], c='g', marker='o', s=300, alpha=0.5)
        return label_


if __name__ == '__main__':
    from sklearn import datasets
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    from sklearn import metrics
    # % matplotlib  inline

    X, y = datasets.make_blobs(n_samples=300, centers=4)
    plt.figure(figsize=(4, 4))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
    kmeans = KMeans(n_clusters=4, max_iter=300)
    kmeans.fit(X_train)
    Y_pred = kmeans.predict(X_test)
    print('acc:{}'.format(metrics.silhouette_score(X_test, Y_pred)))
    del kmeans
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X_train)
    Y_pred = kmeans.predict(X_test)
    print('acc:{}'.format(metrics.silhouette_score(X_test, Y_pred)))