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
from sklearn.cluster import Birch # 引入类模型
from mpl_toolkits.mplot3d import Axes3D

'''去掉特征BEV_chains_mile_std'''
def cal(item1,features_label,data,x,accuracy,si_scores):
    acc_0=[]
    # si_scores=[]
    if(x==item1.shape[1]):
        if data == []:
           return 0
        # combination,acc_0,si_score,accuracy,si_scores=k_means(data,accuracy,si_scores)
        combination, acc_0, si_score, accuracy, si_scores=gmm(data,accuracy,si_scores)
        # combination, acc_0, si_score,accuracy,si_scores=hierarchical(data,accuracy,si_scores)
        # hierarchical(data,accuracy,si_scores)
        # combination, acc_0, si_score, accuracy, si_scores=birch(data,accuracy,si_scores)
        data1=np.array(data)
        # dbscan(data1.T)
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
    acc_0 = accuracy_0(label)
    acc_1 = accuracy_1(label)
    for i in range(len(data)):
        combination.append(get_keys(features_label, data[i].tolist()))
    print('feature combination: {}'.format(combination))
    print('accuracy_0: {}'.format(acc_0))
    print('accuracy_1: {}'.format(acc_1))
    # 使用轮廓系数评估模型的优虐
    si_score = silhouette_score(data.T, label,
                                metric='euclidean', sample_size=len(data.T))
    print('si_score: {:.4f}'.format(si_score))
    accuracy.append(acc_0)
    si_scores.append(si_score)
    return combination, acc_0, si_score,accuracy,si_scores

'dbsacn'
# def dbscan(data):
#     y_pred = DBSCAN(eps=1, min_samples =3).fit_predict(data)
#     plt.scatter(data[:, 0], data[:, 1], c=y_pred)
#     plt.show()

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
    acc_0 = accuracy_0(label)
    acc_1 = accuracy_1(label)
    for i in range(len(data)):
        combination.append(get_keys(features_label, data[i].tolist()))
    print('feature combination: {}'.format(combination))
    print('accuracy_0: {}'.format(acc_0))
    print('accuracy_1: {}'.format(acc_1))
    # 使用轮廓系数评估模型的优虐
    si_score = silhouette_score(data.T, kmeans.labels_,
    metric='euclidean', sample_size=len(data.T))
    print('si_score: {:.4f}'.format(si_score))
    accuracy.append(acc_0)
    si_scores.append(si_score)
    # draw(data, label)
    return combination,acc_0,si_score,accuracy,si_scores

'hierarchical clustering'
def hierarchical(data,accuracy,si_scores):
    combination = []
    data = np.array(data)
    # data=data.T
    # clust = cluster.AgglomerativeClustering(n_clusters=2, linkage='ward')
    clust = cluster.AgglomerativeClustering(n_clusters=2)
    label = clust.fit_predict(data.T)
    print(label)
    label = list(label)
    acc_0 = accuracy_0(label)
    acc_1 = accuracy_1(label)
    for i in range(len(data)):
        combination.append(get_keys(features_label, data[i].tolist()))
    print('feature combination: {}'.format(combination))
    print('accuracy_0: {}'.format(acc_0))
    print('accuracy_1: {}'.format(acc_1))
    # 使用轮廓系数评估模型的优虐
    si_score = silhouette_score(data.T, label,
                                metric='euclidean', sample_size=len(data.T))
    print('si_score: {:.4f}'.format(si_score))
    accuracy.append(acc_0)
    si_scores.append(si_score)
    return combination, acc_0, si_score,accuracy,si_scores


'''Birch'''
def birch(data,accuracy,si_scores):
    combination = []
    data = np.array(data)
    brc = Birch(n_clusters=None).fit(data.T)  # 初始化模型并训练
    print(brc.labels_)
    label = list(brc.labels_)
    acc_0 = accuracy_0(label)
    acc_1 = accuracy_1(label)
    for i in range(len(data)):
        combination.append(get_keys(features_label, data[i].tolist()))
    print('feature combination: {}'.format(combination))
    print('accuracy_0: {}'.format(acc_0))
    print('accuracy_1: {}'.format(acc_1))
    # 使用轮廓系数评估模型的优虐
    si_score = silhouette_score(data, label,
                                metric='euclidean', sample_size=len(data.T))
    print('si_score: {:.4f}'.format(si_score))
    accuracy.append(acc_0)
    si_scores.append(si_score)

    # n_clusters=None是直接将一次遍历建立的特征树作为聚类结果
    # print(brc.root_)  # 特征树根结点
    # print(brc.root_.subclusters_)  # 查看根结点有哪些聚类蔟
    # print(brc.labels_)  # 读取平铺式聚类结果
    return combination, acc_0, si_score, accuracy, si_scores


# def draw(data, label):
#     # 画图
#     fig = plt.figure(figsize=(6, 6))  # 建立画布
#     ax = fig.add_subplot(111, polar=True)  # 增加子网格，注意polar参数
#     labels = np.array(merge_data1.index)  # 设置要展示的数据标签
#     cor_list = ['g', 'r', 'y', 'b']  # 定义不同类别的颜色
#     angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)  # 计算各个区间的角度
#     angles = np.concatenate((angles, [angles[0]]))  # 建立相同首尾字段以便于闭合
#     # 画雷达图
#     for i in range(len(data)):  # 循环每个类别
#         data_tmp = num_sets_max_min[i, :]  # 获得对应类数据
#         data = np.concatenate((data_tmp, [data_tmp[0]]))  # 建立相同首尾字段以便于闭合
#         ax.plot(angles, data, 'o-', c=cor_list[i], label="第%d类渠道" % (i))  # 画线
#         ax.fill(angles, data, alpha=2.5)
#     # 设置图像显示格式
#     ax.set_thetagrids(angles * 180 / np.pi, labels, fontproperties="SimHei")  # 设置极坐标轴
#     ax.set_title("各聚类类别显著特征对比", fontproperties="SimHei")  # 设置标题放置
#     ax.set_rlim(-0.2, 1.2)  # 设置坐标轴尺度范围
#     plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))  # 设置图例位置


def accuracy_0(label):
    n=0
    m=0
    label_0=list([0,0,0,0,0,1,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0])
    for i in label:
        if i == label_0[n] :
            m = m + 1
        n=n+1
    acc_0=m/24
    return acc_0

def accuracy_1(label):
    m=0
    n=0
    label_1=list([0,1,1,0,0,2,1,2,2,0,2,1,0,0,0,2,1,1,0,0,1,0,1,1])
    for i in label:
        if i == label_1[n]:
            m = m + 1
        n=n+1
    acc_1=m/24
    return acc_1

if __name__ == '__main__':
    accuracy=[]
    si_scores=[]
    '''BEV特征提取'''
#     BEV_daily_distance_mean =[-1.27297608e+00,-3.31186525e-01,-8.88199534e-01,1.58553896e-01
# ,6.79238895e-01,-1.16244969e+00,-6.93371124e-01,-1.42996287e+00
# ,1.81675633e+00,-1.04658297e+00,-1.15622421e+00,3.06880465e-01
# ,7.85185089e-01,1.33681523e+00,9.64807655e-01,1.06957509e+00
# ,-1.35981619e+00,-4.15650753e-01,-5.55837639e-01,1.96829085e+00
# ,-5.79848260e-02,3.33180935e-01,9.45355710e-01,-5.44518662e-01
# ,5.50120927e-01]# 统计日均里程
#     BEV_daily_distance_std= [-1.43304586e+00,-1.73399840e-01,-5.19034126e-01,-2.55196290e-01
# ,1.14662441e+00,-8.92082059e-01,-7.23329845e-02,-1.17480768e+00
# ,9.35550333e-01,-1.20625860e+00,-8.66588667e-01,6.00660236e-01
# ,1.16592168e+00,1.63898494e+00,1.55671494e+00,1.42858158e+00
# ,-1.92664183e+00,-1.47391442e-01,1.07528687e-01,4.62540607e-01
# ,3.72993468e-01,-6.92609454e-01,1.22294488e+00,-9.87338532e-01
# ,-2.92318396e-01]
#     BEV_night_distance_mean= [-9.66893762e-01,-8.65179220e-01,2.23380754e-01,1.32342116e+00
# ,1.17644203e+00,-3.03573204e-02,-3.50913402e-01,-9.17379256e-01
# ,8.08585702e-02,-7.99328488e-01,-3.63002597e-01,-3.20844345e-01
# ,-5.77354186e-01,-4.75119037e-01,2.07220995e+00,2.93229346e+00
# ,-1.06353554e+00,-1.43337473e-01,3.23857076e-01,9.61372544e-01
# ,-8.32171844e-01,-1.18315280e+00,4.83705565e-01,-1.32969181e-01
# ,-5.56002658e-01]
#     BEV_night_distance_percentage=[-1.15819501e+00,-5.18249399e-01,1.84615395e+00,9.84869766e-01
# ,8.78007120e-01,-1.04406825e+00,4.93896665e-01,-4.89867986e-01
# ,-9.27860884e-01,-3.00765674e-01,7.38708552e-02,1.77185901e-01
# ,-8.89851997e-01,-7.92397447e-01,1.97645595e+00,1.81333355e+00
# ,-1.10324875e+00,-6.51177768e-01,1.32216344e+00,9.29652453e-01
# ,-7.92415659e-01,-1.19625322e+00,4.18776090e-01,-6.37262511e-01
# ,-4.12751182e-01]
#     BEV_am_peak_percentage= [1.56209441e+00,-2.26387631e-01,-4.19282866e-01,-1.33250481e+00
# ,-6.11503740e-01,1.83062600e+00,-2.49819178e-01,-3.86758266e-01
# ,-1.35152925e-01,-8.01422861e-02,1.10763790e-01,-1.39838054e+00
# ,-2.79757091e-01,-1.21672571e+00,-4.41333129e-01,-4.83515932e-01
# ,6.47223976e-01,1.24877458e-01,-3.22718537e-01,-6.24734529e-01
# ,1.95114485e+00,1.66594861e+00,-7.99052751e-01,1.94840213e+00
# ,-8.33311305e-01]
#     BEV_pm_peak_percentage=[1.50902327e+00,-6.94023762e-01,-2.54840166e-01,5.67481455e-01
# ,-7.11127384e-01,-2.26585767e-01,-3.76496550e-01,-7.47975956e-01
# ,-1.25423353e-01,-6.30935879e-01,5.90973782e-01,9.98502590e-01
# ,-5.29198402e-02,4.76224188e-01,-2.18872051e-01,-2.16573224e+00
# ,3.02263648e+00,-2.33829599e-01,-7.06576115e-01,-8.00442115e-02
# ,8.07929658e-01,1.15191573e+00,-2.38681054e-01,-2.15554574e-01
# ,-1.44506865e+00]
#     BEV_weekends_distance_percentage= [1.55202111e-01,-9.51343641e-01,1.60237007e+00,-6.84003091e-02
# ,-1.25628785e-01,-2.30603038e+00,1.76283564e+00,1.42114768e+00
# ,7.02634412e-01,-9.23011997e-01,7.44169302e-01,-8.70593717e-01
# ,2.35202053e-03,-1.05366128e+00,4.18955838e-01,-9.35635720e-01
# ,-6.74031732e-01,-1.57720478e-01,1.46979387e+00,6.94523562e-01
# ,-7.72554133e-01,4.72091696e-01,3.56932411e-01,-1.40015079e+00
# ,4.35754341e-01]
#     BEV_charging_rate_mean=[-1.00775227e+00,-7.56950533e-01,-3.62597845e-01,1.31137738e+00
# ,7.04633601e-01,-7.02342073e-01,-9.14806203e-01,-7.97729157e-01
# ,2.34761719e+00,-9.79710939e-01,-9.90983866e-01,-3.67490926e-01
# ,9.24492064e-01,1.65814429e+00,7.95534215e-01,1.27399335e+00
# ,-7.20069312e-01,-7.62252063e-01,-8.61904527e-01,9.30048972e-01
# ,-1.02203772e+00,7.66278662e-01,7.34104745e-01,-7.60062400e-01
# ,-4.39534628e-01]
#     BEV_chains_num=[-1.27261757e+00,-3.55873724e-01,2.30253171e-01,-3.02621297e-01
# ,7.71132320e-01,-1.24149953e+00,3.64045641e-01,-1.91504527e+00
# ,9.64589484e-01,-1.11262297e+00,-1.24801396e+00,4.00464977e-02
# ,4.53664888e-01,3.84247412e-01,7.48131752e-01,1.28342453e+00
# ,-2.30509883e+00,5.05552698e-01,6.65600449e-01,1.49715289e+00
# ,1.14131570e-01,5.49690090e-01,1.24515224e+00,-6.49898200e-01
# ,5.86475718e-01]
#     BEV_chains_mile_std= [-3.30812936e-01,-2.68602887e-01,-2.98512353e-01,-1.11023339e-01
# ,-1.79594645e-01,-2.11033937e-01,-2.58966958e-01,-3.81251647e-01
# ,1.92968343e-01,-3.29791332e-01,-2.18978506e-01,-3.40293264e-02
# ,-2.22989675e-01,6.16346170e-02,-1.96321155e-01,-1.05363341e-01
# ,4.86017930e+00,-2.11345754e-01,-2.91584659e-01,-1.40714788e-01
# ,-2.25438649e-01,-2.97200112e-01,-2.25128585e-01,-3.26049654e-01
# ,-2.50048022e-01]
#     BEV_daily_num_run_status= [-9.78906481e-01,-1.12917846e-01,-9.20045368e-01,7.93286202e-01
# ,8.08756844e-01,-1.13313074e+00,-7.58008220e-01,-1.19837453e+00
# ,6.12000142e-01,-8.78363336e-01,-1.09078947e+00,2.42162670e-01
# ,1.40588053e+00,5.34432337e-01,1.82441731e+00,5.79195295e-01
# ,-1.16639048e+00,-7.08340923e-01,-7.09025437e-01,2.30080132e+00
# ,-5.53001216e-01,3.70889018e-03,1.01022074e+00,-7.58327657e-01
# ,8.50759420e-01]

    '------------------------------------------'
    BEV_daily_distance_mean=[0.04619631,0.32333558,0.15942404,0.4674509,0.62067225,0.07872078
,0.21675596,0,0.95540812,0.11281674,0.08055274,0.51109878
,0.6518489,0.81417644,0.70470622,0.73553601,0.02064198,0.2984804
,0.25722777,1,0.40373032,0.51883819,0.69898212,0.26055859
,0.58267686]
    BEV_daily_distance_std=[0.13843175,0.49170654,0.39477146,0.46876626,0.86191473,0.29014808
,0.5200513,0.2108561,0.80271782,0.20203551,0.29729785,0.70879602
,0.86732676,1,0.97692692,0.9409912,0,0.49900074
,0.57049451,0.6700596,0.6449456,0.34609129,0.88331923,0.26343287
,0.45835516]
    BEV_night_distance_mean=[0.05254814,0.07726345,0.34176939,0.60906492,0.5733509,0.28011433
,0.20222337,0.06457952,0.30713835,0.09326432,0.19928585,0.20952976
,0.1472012,0.17204301,0.79101088,1,0.02906544,0.25266162
,0.36618383,0.52109181,0.08528382,0,0.40502494,0.25518098
,0.15238934]
    BEV_night_distance_percentage=[0.01199549,0.2136987,0.95893037,0.68746389,0.65378206,0.04796688
,0.53271504,0.22264418,0.08459406,0.28224697,0.40032792,0.43289159
,0.096574,0.12729051,1,0.94858577,0.0293139,0.17180127
,0.79377482,0.67006005,0.12728477,0,0.50903793,0.17618719
,0.24695048]
    BEV_am_peak_percentage=[0.88384909,0.3498982,0.29230938,0.01966718,0.23492188,0.96401913
,0.34290272,0.30201959,0.3771363,0.39355971,0.45055468,0.
,0.33396476,0.05423301,0.28572627,0.27313261,0.61071474,0.45476831
,0.32113863,0.23097183,1,0.91485473,0.17892917,0.99918116
,0.16870128]
    BEV_pm_peak_percentage=[0.708268,0.28365534,0.36830306,0.52679635,0.28035881,0.37374878
,0.34485515,0.27325666,0.3932467,0.29581482,0.53132423,0.60987085
,0.40722094,0.50920753,0.37523551,0,1,0.37235261
,0.28123601,0.40199302,0.57314005,0.63943951,0.37141755,0.37587492
,0.13889984]
    BEV_weekends_distance_percentage=[0.60489396,0.33293963,0.96056258,0.54993948,0.53587451,0.
,1,0.91602379,0.7394357,0.33990266,0.74964368,0.35278543
,0.56732819,0.30779315,0.66971638,0.33680014,0.40109422,0.52798738
,0.9279795,0.73744231,0.3768805,0.68277551,0.65447296,0.22263687
,0.67384493]
    BEV_charging_rate_mean=[0.00423944,0.07866894,0.19569953,0.69247895,0.51241785,0.09487489
,0.0318227,0.06656722,1,0.01256116,0.00921574,0.19424743
,0.57766443,0.79538768,0.53939409,0.68138463,0.08961405,0.07709563
,0.04752213,0.57931353,0,0.53071203,0.52116389,0.07774545
,0.17286728]
    BEV_chains_num=[0.27154469,0.5126502,0.66680278,0.5266557,0.8090551,0.2797288
,0.70199047,0.10258489,0.85993473,0.3136236,0.27801549,0.61677803
,0.72556052,0.70730358,0.8030059,0.943789,0,0.73920712
    ,0.7813,1,0.63626256,0.75081534,0.93372331,0.43532116
    ,0.76049004]
    BEV_chains_mile_std=[0.00962308,0.02149199,0.01578563,0.05155621,0.03847365,0.03247543
,0.0233304,0,0.10955405,0.00981799,0.0309597,0.06624571
,0.03019442,0.0844972,0.03528244,0.05263607,1,0.03241594
,0.01710735,0.04589145,0.02972719,0.01603599,0.02978634,0.01053186
,0.02503202]
    BEV_daily_num_run_status=[0.06271993,0.31020353,0.07954135,0.56917995,0.57360117,0.01864548
,0.12584858,0,0.51737173,0.0914533,0.03074583,0.411679
,0.74424812,0.49520428,0.86385823,0.50799671,0.00914045,0.14004258
,0.13984696,1,0.18443581,0.3435333,0.63117584,0.12575729
,0.58560474]

    # BEV_daily_num_run_status=np.delete(BEV_daily_num_run_status,16,axis=0)


    # features = np.array([BEV_daily_distance_mean, \
    #               BEV_daily_distance_std, \
    #               BEV_night_distance_mean, \
    #               BEV_night_distance_percentage, \
    #               BEV_am_peak_percentage, \
    #               BEV_pm_peak_percentage, \
    #               BEV_weekends_distance_percentage, \
    #               BEV_charging_rate_mean,
    #               BEV_chains_num, \
    #               # BEV_chains_mile_std, \
    #               BEV_daily_num_run_status,\
    #               ])

    features = [BEV_daily_distance_mean, \
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
                         ]

    features_1=[]
    for i in features:
        features_1.append(np.delete(i,16,axis=0))

    features_1=np.array(features_1)

    features_1_label = {0: features_1[0],
                      1: features_1[1],
                      2: features_1[2],
                      3: features_1[3],
                      4: features_1[4],
                      5: features_1[5],
                      6: features_1[6],
                      7: features_1[7],
                      8: features_1[8],
                      9: features_1[9],
                      10: features_1[10]
                      }


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
                     # 9: BEV_daily_num_run_status
                     10: BEV_daily_num_run_status
                      }

    # 计算每一列的平均值
    # meandata = np.mean(features, axis=0)
    # 均值归一化
    # features = features - meandata
    # 求协方差矩阵
    # cov = np.cov(features.T.transpose())
    # # 求解特征值和特征向量
    # eigVals, eigVectors = np.linalg.eig(cov)
    # # 选择前两个特征向量
    # pca_mat = eigVectors[:, :3]
    # pca_data = np.dot(features.T, pca_mat)
    # pca_data = pd.DataFrame(pca_data, columns=['pca1', 'pca2', 'pca3'])
    #
    # # 两个主成分的散点图
    # plt.subplot(111)
    # plt.scatter(pca_data['pca1'], pca_data['pca2'], pca_data['pca3'])
    # plt.xlabel('pca_1')
    # plt.ylabel('pca_2')
    # plt.ylabel('pca_3')
    # plt.show()
    # # print('前两个主成分包含的信息百分比：{:.2%}'.format(np.sum(eigVals[:2]) / np.sum(eigVals)))
    # print('前3个主成分包含的信息百分比：{}'.format(np.sum(eigVals[:3]) / np.sum(eigVals)))
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # For each set of style and range settings, plot n random points in the box
    # # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    # for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    #     # xs = randrange(n, 23, 32)
    #     # ys = randrange(n, 0, 100)
    #     # zs = randrange(n, zlow, zhigh)
    #     # ax.scatter(xs, ys, zs, c=c, marker=m)
    #     # ax.scatter(pca_data['pca1'], pca_data['pca2'], pca_data['pca3'], label='parametric curve')
    #     ax.scatter(pca_data['pca1'], pca_data['pca2'], pca_data['pca3'], c=c,marker=m)
    # plt.show()

    features_1_label = [features_1_label[i] for i in range(len(features))]
    features_1_label = np.array(features_1_label)
    acc_0, si_score, accuracy = cal(features_1_label.T, features_1_label, [], 0, accuracy, si_scores)
    print('----------------------------------------')
    print('The best accuracy: {}'.format(findBestAccuracy(accuracy)))
    print('----------------------------------------')
    print('The best si_score: {}'.format(findBestSi(si_score)))


    # feature_label = [features_label[i] for i in range(len(features))]
    # feature_label=np.array(feature_label)
    # acc_0,si_score,accuracy=cal(feature_label.T,features_label,[],0,accuracy,si_scores)
    # print('----------------------------------------')
    # print('The best accuracy: {}'.format(findBestAccuracy(accuracy)))
    # print('----------------------------------------')
    # print('The best si_score: {}'.format(findBestSi(si_score)))




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

