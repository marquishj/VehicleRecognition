# -*- coding: utf-8 -*-
# @Time    : 2020/10/28 21:45
# @Author  : Hou Jue
# @Email   : 382062104@qq.com
# @File    : cxh_K_means_1.py
# @Software: PyCharm

import numpy as np
from sklearn.cluster import KMeans
# import sklearn.preprocessing.MinMaxScaler
# import sklearn.preprocessing.StandardScaler
# import sklearn
from sklearn import preprocessing

data=np.array([[3.8,31.3,76.5,66.3,217.7,133.9],
               [3.7,16.4,28.0,25.6,55.6,52.3],
                [21.5,19.8,79.0,59.1,216.6,156.3],
                [18.1,15.8,34.9,26.1,51.2,52.1],
                [1.4,1.3,2.9,2.3,9.4,5.7],\
                [0.7,0.6,1.7,0.8,2.6,2.5],\
               [ 0.9,0.9,2.8,2.1,9.1,5.5],
               [ 0.7,0.6,1.4,0.9,2.5,2.3],
               [ 5.6,5.5,8.3,7.4,20.6,12.5],
               [ 2.4,2.3,3.1,3.2,4.9,5.5],
               [ 0.4,0.4,0.3,0.3,0.1,0.2],
              [  0.1,0.1,0.1,0.1,0.03,0.1],
               [ 2.8,2.7,3.6,3.3,5.6,4.3],
               [ 0.7,0.7,0.8,0.7,0.9,0.8],
               [ 1.5,1.6,1.8,1.9,2.6,2.1],
               [ 0.5,0.5,0.6,0.6,0.9,0.7],
                [1.2,1.0,0.9,0.7,0.9,0.8],
               [ 1.1,1.1,1.1,0.8,0.7,0.7],
                [0.6,0.6,0.3,0.3,0.6,0.4],
               [ 0.8,0.7,0.8,0.7,0.8,0.6]])
BEV=[data[:,0],data[:,2],data[:,4]]
PHEV=[data[:,1],data[:,3],data[:,5]]

BEV=np.array(BEV)
PHEV=np.array(PHEV)


kmeans = KMeans(
    n_clusters=3,  # 簇的个数，默认值为8
    random_state=0
).fit(BEV)
label = kmeans.labels_
print(label)
