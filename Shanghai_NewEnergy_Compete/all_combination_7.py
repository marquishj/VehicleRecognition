# -*- coding: utf-8 -*-
# @Time    : 2020/10/27 16:49
# @Author  : Hou Jue
# @Email   : 382062104@qq.com
# @File    : all_combination_3.py
# @Software: PyCharm
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing



# -*- coding: utf-8 -*-
'''a=XX'''
# globals(n)
def sub_sets(a,b,idx):
    # n = n + 1
    if(idx==len(a)):
        pass
        # n=n+1
        # print(b)
        # print(XX[i] for i in a)

    else:
        c=b[:]
        b.append(a[idx])
        sub_sets(a,b,idx+1)
        sub_sets(a,c,idx+1)
        k_means(b)




def k_means(b):
    kmeans = KMeans(
        n_clusters=2,  # 簇的个数，默认值为8
        random_state=0
    ).fit(b)

    print(kmeans.labels_)
    label=kmeans.labels_
    m=accuracy(label)
    print('accuracy{}'.format(m))
    # print("K Clusters Centroids:\n", kmeans.cluster_centers_)


def accuracy(label):
    n=0
    label_0=[0,0,0,0,0,1,0,1,1,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0]
    label_1=[0,1,1,0,0,2,1,2,2,0,2,1,0,0,0,2,2,1,1,0,0,1,0,1,1]
    for i in range(25):
        for j in range(25):
            if label[i] == label_0[j]:
                n=n+1
    m=n/25*100
    return m

if __name__ == '__main__':

    '''BEV特征提取'''
    BEV_daily_distance_mean = [0,0]  # 统计日均里程
    BEV_daily_distance_std = [1,1]  # 统计日里程标准差
    BEV_night_distance_mean = [2,2]  ##统计夜间日均里程
    BEV_night_distance_percentage = [3,3]  # 统计夜间里程占比 0：00-5：00
    BEV_am_peak_percentage = [4,4]  # 统计早高峰里程占比  7：00-9：00
    BEV_pm_peak_percentage = [5,5]  # 统计晚高峰里程占比  17：00-19：00
    BEV_weekends_distance_percentage = [6,6]  # 统计周末里程占比
    BEV_charging_rate_mean = [7,7]  # 统计充电速率



    X = np.array([BEV_daily_distance_mean, \
                  BEV_daily_distance_std, \
                  BEV_night_distance_mean, \
                  BEV_night_distance_percentage, \
                  BEV_am_peak_percentage, \
                  BEV_pm_peak_percentage, \
                  BEV_weekends_distance_percentage, \
                  BEV_charging_rate_mean])

    '''归一化'''
    min_max_scaler = preprocessing.MinMaxScaler()

    X_minMax = min_max_scaler.fit_transform(X)

    BEV_daily_distance_mean=min_max_scaler.fit_transform(BEV_daily_distance_mean)
    BEV_daily_distance_std=min_max_scaler.fit_transform(BEV_daily_distance_std)
    BEV_night_distance_mean=min_max_scaler.fit_transform(BEV_night_distance_mean)
    BEV_night_distance_percentage=min_max_scaler.fit_transform(BEV_night_distance_percentage)
    BEV_am_peak_percentage=min_max_scaler.fit_transform(BEV_am_peak_percentage)
    BEV_pm_peak_percentage=min_max_scaler.fit_transform(BEV_pm_peak_percentage)
    BEV_weekends_distance_percentage=min_max_scaler.fit_transform(BEV_weekends_distance_percentage)
    BEV_charging_rate_mean=min_max_scaler.fit_transform(BEV_charging_rate_mean)




    XX = {0: BEV_daily_distance_mean,
          1: BEV_daily_distance_std,
          2: BEV_night_distance_mean,
          3: BEV_night_distance_percentage,
          4: BEV_am_peak_percentage,
          5: BEV_pm_peak_percentage,
          6: BEV_weekends_distance_percentage,
          7: BEV_charging_rate_mean,
          }

    X_key = [XX[i] for i in range(8)]
    X_key = np.array(X_key)
    # sub_sets([1,2,3],[],0)
    # sub_sets(X_key,[],0)
    sub_sets(X_key.T,[],0)