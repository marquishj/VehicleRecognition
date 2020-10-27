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
def sub_sets(a, b, idx):
    # n = n + 1
    if (idx == len(a)):
        pass
        # n=n+1
        # print(b)
        # print(XX[i] for i in a)

    else:
        c = b[:]
        b.append(a[idx])
        sub_sets(a, b, idx + 1)
        sub_sets(a, c, idx + 1)
        k_means(b)


def k_means(b):
    kmeans = KMeans(
        n_clusters=2,  # 簇的个数，默认值为8
        random_state=0
    ).fit(b)

    # print(kmeans.labels_)
    label = kmeans.labels_
    label = list(label)
    print(label)
    acc_0 = accuracy_0(label)
    acc_1 = accuracy_1(label)
    print('accuracy_0: {}'.format(acc_0))
    print('accuracy_1: {}'.format(acc_1))
    # print("K Clusters Centroids:\n", kmeans.cluster_centers_)


def accuracy_0(label):
    n = 0
    label_0 = list([0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    # label=list[label]
    # print(label)
    for i in label:
        # for i in range(20):
        if i == label_0[i]:
            n = n + 1
    #     continue
    # if label[1] == label_0[1]:
    #     n=n+1
    acc_0 = n / 25
    return acc_0


def accuracy_1(label):
    m = 0
    label_1 = [0, 1, 1, 0, 0, 2, 1, 2, 2, 0, 2, 1, 0, 0, 0, 2, 2, 1, 1, 0, 0, 1, 0, 1, 1]
    # for i in range(len(label)):
    for i in label:
        if i == label_1[i]:
            m = m + 1
    acc_1 = m / 25
    return acc_1


if __name__ == '__main__':
    '''BEV特征提取'''

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

    # BEV_daily_distance_mean=min_max_scaler.fit_transform(BEV_daily_distance_mean)
    # BEV_daily_distance_std=min_max_scaler.fit_transform(BEV_daily_distance_std)
    # BEV_night_distance_mean=min_max_scaler.fit_transform(BEV_night_distance_mean)
    # BEV_night_distance_percentage=min_max_scaler.fit_transform(BEV_night_distance_percentage)
    # BEV_am_peak_percentage=min_max_scaler.fit_transform(BEV_am_peak_percentage)
    # BEV_pm_peak_percentage=min_max_scaler.fit_transform(BEV_pm_peak_percentage)
    # BEV_weekends_distance_percentage=min_max_scaler.fit_transform(BEV_weekends_distance_percentage)
    # BEV_charging_rate_mean=min_max_scaler.fit_transform(BEV_charging_rate_mean)

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
    # sub_sets([1,2,3],[],0)
    X_key = np.array(X_key)
    sub_sets(X_key.T, [], 0)
    print(n)