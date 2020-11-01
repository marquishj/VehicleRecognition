# -*- coding: utf-8 -*-
# @Time    : 2020/10/27 16:49
# @Author  : Hou Jue
# @Email   : 382062104@qq.com
# @File    : all_combination_3.py
# @Software: PyCharm
# -*- coding: utf-8 -*-
# import numpy as np
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
# import sklearn.preprocessing.MinMaxScaler
# import sklearn.preprocessing.StandardScaler
# import sklearn
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
# -*- coding: utf-8 -*-
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn import cluster
from sklearn.preprocessing import StandardScaler

def cal(item1,features_label,data,x,accuracy,si_scores):
    acc_0=[]
    # si_scores=[]
    if(x==item1.shape[1]):
        if data == []:
           return 0
        # combination,acc_0,si_score,accuracy,si_scores=k_means(data,accuracy,si_scores)
        # combination, acc_0, si_score, accuracy, si_scores=gmm(data,accuracy,si_scores)
        combination, acc_0, si_score,accuracy,si_scores=hierarchical(data,accuracy,si_scores)
    else:
        item3=data[:]
        data.append(item1[:,x])
        cal(item1,features_label,data,x+1,accuracy,si_scores)
        cal(item1,features_label,item3,x+1,accuracy,si_scores)
    return acc_0,si_scores,accuracy

def get_keys(features_label, value):
    return [k for k, v in features_label.items() if v == value]

def findBestAccuracy(accuracy):
    bestAccuracy=max(accuracy)

    return bestAccuracy

def findBestSi(si_scores):
    bestSi_score=max(si_scores)
    return bestSi_score

'GMM'
def gmm(data,accuracy,si_scores):
    combination = []
    data = np.array(data)
    # gmmModel = GaussianMixture(n_components=2, covariance_type='diag', random_state=0)
    gmmModel = GaussianMixture(n_components=2, covariance_type='diag', random_state=2)
    gmmModel.fit(data.T)
    label = gmmModel.predict(data.T)
    print(label)
    label = list(label)
    for i in range(len(data)):
        combination.append(get_keys(features_label, data[i].tolist()))
    print('feature combination: {}'.format(combination))
    print('accuracy_0: {}'.format(acc_0))
    # 使用轮廓系数评估模型的优虐
    si_score = silhouette_score(data.T, label,
                                metric='euclidean', sample_size=len(data.T))
    print('si_score: {:.4f}'.format(si_score))
    # accuracy.append(acc_0)
    # si_scores.append(si_score)
    return combination, acc_0, si_score,accuracy,si_scores



'k_means方法'
def k_means(data,accuracy,si_scores):
    combination = []
    data=np.array(data)
    kmeans = KMeans(
        n_clusters=2,
        random_state=9
    ).fit(data.T)
    label = kmeans.labels_
    print(label)
    label = list(label)
    # acc_0 = accuracy_0(label)
    # acc_1 = accuracy_1(label)
    for i in range(len(data)):
        combination.append(get_keys(features_label, data[i].tolist()))
    print('feature combination: {}'.format(combination))

    # 使用轮廓系数评估模型的优虐
    si_score = silhouette_score(data.T, kmeans.labels_,
    metric='euclidean', sample_size=len(data.T))
    print('si_score: {:.4f}'.format(si_score))
    # accuracy.append(acc_0)
    # si_scores.append(si_score)
    # draw(data, label)
    return combination,acc_0,si_score,accuracy,si_scores

'hierarchical clustering'
def hierarchical(data,accuracy,si_scores):
    combination = []
    data = np.array(data)
    clust = cluster.AgglomerativeClustering(n_clusters=2)
    label = clust.fit_predict(data.T)
    print(label)
    label = list(label)
    # acc_0 = accuracy_0(label)
    # acc_1 = accuracy_1(label)
    for i in range(len(data)):
        combination.append(get_keys(features_label, data[i].tolist()))
    print('feature combination: {}'.format(combination))
    # 使用轮廓系数评估模型的优虐
    si_score = silhouette_score(data.T, label,
                                metric='euclidean', sample_size=len(data.T))
    print('si_score: {:.4f}'.format(si_score))
    # accuracy.append(acc_0)
    # si_scores.append(si_score)
    return combination, acc_0, si_score,accuracy,si_scores



if __name__ == '__main__':

    si_scores=[]

    '''BEV特征标准化'''
    scaler = StandardScaler()
    BEV_daily_distance_mean=scaler.fit(BEV_daily_distance_mean)
    BEV_daily_distance_std=scaler.fit(BEV_daily_distance_std)
    BEV_night_distance_mean=scaler.fit(BEV_night_distance_mean)
    BEV_night_distance_percentage=scaler.fit(BEV_night_distance_percentage)
    BEV_am_peak_percentage=scaler.fit(BEV_am_peak_percentage)
    BEV_pm_peak_percentage=scaler.fit(BEV_pm_peak_percentage)
    BEV_weekends_distance_percentage=scaler.fit(BEV_weekends_distance_percentage)
    BEV_charging_rate_mean=scaler.fit(BEV_charging_rate_mean)
    BEV_chains_num=scaler.fit(BEV_chains_num)
    BEV_chains_mile_std=scaler.fit(BEV_chains_mile_std)
    BEV_daily_num_run_status=scaler.fit(BEV_daily_num_run_status)


    PHEV_daily_distance_mean=scaler.fit(PHEV_daily_distance_mean)
    PHEV_daily_distance_std=scaler.fit(PHEV_daily_distance_std)
    PHEV_night_distance_mean=scaler.fit(PHEV_night_distance_mean)
    PHEV_night_distance_percentage=scaler.fit(PHEV_night_distance_percentage)
    PHEV_am_peak_percentage=scaler.fit(PHEV_am_peak_percentage)
    PHEV_pm_peak_percentage=scaler.fit(PHEV_pm_peak_percentage)
    PHEV_weekends_distance_percentage=scaler.fit(PHEV_weekends_distance_percentage)
    PHEV_charging_rate_mean=scaler.fit(PHEV_charging_rate_mean)
    PHEV_chains_num=scaler.fit(PHEV_chains_num)
    PHEV_chains_mile_std=scaler.fit(PHEV_chains_mile_std)
    PHEV_daily_num_run_status=scaler.fit(PHEV_daily_num_run_status)

    '''计算时注释掉另一个即可'''
    features = np.array([BEV_daily_distance_mean, \
                         BEV_daily_distance_std, \
                         BEV_night_distance_mean, \
                         BEV_night_distance_percentage, \
                         BEV_am_peak_percentage, \
                         BEV_pm_peak_percentage, \
                         BEV_weekends_distance_percentage, \
                         BEV_charging_rate_mean,
                         BEV_chains_num, \
                         BEV_chains_mile_std, \
                         BEV_daily_num_run_status, \
                         ])

    features_label = {0: BEV_daily_distance_mean,
                      1: BEV_daily_distance_std,
                      2: BEV_night_distance_mean,
                      3: BEV_night_distance_percentage,
                      4: BEV_am_peak_percentage,
                      5: BEV_pm_peak_percentage,
                      6: BEV_weekends_distance_percentage,
                      7: BEV_charging_rate_mean,
                      8: BEV_chains_num,
                      9: BEV_chains_mile_std,
                     10: BEV_daily_num_run_status,
                     # 10: BEV_daily_num_run_status
                      }

    '''计算时注释掉另一个即可'''
    features = np.array([PHEV_daily_distance_mean, \
                         PHEV_daily_distance_std, \
                         PHEV_night_distance_mean, \
                         PHEV_night_distance_percentage, \
                         PHEV_am_peak_percentage, \
                         PHEV_pm_peak_percentage, \
                         PHEV_weekends_distance_percentage, \
                         PHEV_charging_rate_mean,
                         PHEV_chains_num, \
                         PHEV_chains_mile_std, \
                         PHEV_daily_num_run_status, \
                         ])
    features_label = {0: PHEV_daily_distance_mean,
                      1: PHEV_daily_distance_std,
                      2: PHEV_night_distance_mean,
                      3: PHEV_night_distance_percentage,
                      4: PHEV_am_peak_percentage,
                      5: PHEV_pm_peak_percentage,
                      6: PHEV_weekends_distance_percentage,
                      7: PHEV_charging_rate_mean,
                      8: PHEV_chains_num,
                      9: PHEV_chains_mile_std,
                      10: PHEV_daily_num_run_status,
                      }



    feature_label = [features_label[i] for i in range(len(features))]
    feature_label=np.array(feature_label)
    acc_0,si_score,accuracy=cal(feature_label.T,features_label,[],0,[],si_scores)
    print('----------------------------------------')
    print('The best si_score: {}'.format(findBestSi(si_score)))




    #
    # score_list = list()
    # silhouette_int = -1
    # for n_clusters in range(2,10):
    #     model_kmeans = KMeans(n_clusters=n_clusters)  #建立聚类模型对象
    #     labels_tmp = model_kmeans.fit_predict(features.T)  #训练聚类模型
    #     silhouette_tmp = silhouette_score(features.T, labels_tmp)  #得到每个K下的平均轮廓系数
    #     if silhouette_tmp > silhouette_int:  #如果平均z轮廓系数更高
    #         best_k = n_clusters  #保存最佳的K值
    #         silhouette_int = silhouette_tmp
    #         best_kmeans = model_kmeans  #保存模型实例对象
    #         cluster_labels_k = labels_tmp  #保存聚类标签
    #     score_list.append([n_clusters, silhouette_tmp])  #将每次K值及其平均轮廓系数记录

    # print(score_list)
    # print('最优的K值是：{0} \n对应的轮廓系数是{1}'.format(best_k, silhouette_int))

