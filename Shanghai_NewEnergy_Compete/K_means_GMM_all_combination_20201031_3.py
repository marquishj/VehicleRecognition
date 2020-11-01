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
    accuracy.append(acc_0)
    si_scores.append(si_score)
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
    acc_0 = accuracy_0(label)
    acc_1 = accuracy_1(label)
    for i in range(len(data)):
        combination.append(get_keys(features_label, data[i].tolist()))
    print('feature combination: {}'.format(combination))

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
    clust = cluster.AgglomerativeClustering(n_clusters=2)
    label = clust.fit_predict(data.T)
    print(label)
    label = list(label)
    acc_0 = accuracy_0(label)
    acc_1 = accuracy_1(label)
    for i in range(len(data)):
        combination.append(get_keys(features_label, data[i].tolist()))
    print('feature combination: {}'.format(combination))
    # 使用轮廓系数评估模型的优虐
    si_score = silhouette_score(data.T, label,
                                metric='euclidean', sample_size=len(data.T))
    print('si_score: {:.4f}'.format(si_score))
    accuracy.append(acc_0)
    si_scores.append(si_score)
    return combination, acc_0, si_score,accuracy,si_scores



if __name__ == '__main__':
    accuracy=[]
    si_scores=[]
    '''BEV特征提取'''
    BEV_daily_distance_mean =[-1.27297608e+00,-3.31186525e-01,-8.88199534e-01,1.58553896e-01
,6.79238895e-01,-1.16244969e+00,-6.93371124e-01,-1.42996287e+00
,1.81675633e+00,-1.04658297e+00,-1.15622421e+00,3.06880465e-01
,7.85185089e-01,1.33681523e+00,9.64807655e-01,1.06957509e+00
,-1.35981619e+00,-4.15650753e-01,-5.55837639e-01,1.96829085e+00
,-5.79848260e-02,3.33180935e-01,9.45355710e-01,-5.44518662e-01
,5.50120927e-01]# 统计日均里程
    BEV_daily_distance_std= [-1.43304586e+00,-1.73399840e-01,-5.19034126e-01,-2.55196290e-01
,1.14662441e+00,-8.92082059e-01,-7.23329845e-02,-1.17480768e+00
,9.35550333e-01,-1.20625860e+00,-8.66588667e-01,6.00660236e-01
,1.16592168e+00,1.63898494e+00,1.55671494e+00,1.42858158e+00
,-1.92664183e+00,-1.47391442e-01,1.07528687e-01,4.62540607e-01
,3.72993468e-01,-6.92609454e-01,1.22294488e+00,-9.87338532e-01
,-2.92318396e-01]
    BEV_night_distance_mean= [-9.66893762e-01,-8.65179220e-01,2.23380754e-01,1.32342116e+00
,1.17644203e+00,-3.03573204e-02,-3.50913402e-01,-9.17379256e-01
,8.08585702e-02,-7.99328488e-01,-3.63002597e-01,-3.20844345e-01
,-5.77354186e-01,-4.75119037e-01,2.07220995e+00,2.93229346e+00
,-1.06353554e+00,-1.43337473e-01,3.23857076e-01,9.61372544e-01
,-8.32171844e-01,-1.18315280e+00,4.83705565e-01,-1.32969181e-01
,-5.56002658e-01]
    BEV_night_distance_percentage=[-1.15819501e+00,-5.18249399e-01,1.84615395e+00,9.84869766e-01
,8.78007120e-01,-1.04406825e+00,4.93896665e-01,-4.89867986e-01
,-9.27860884e-01,-3.00765674e-01,7.38708552e-02,1.77185901e-01
,-8.89851997e-01,-7.92397447e-01,1.97645595e+00,1.81333355e+00
,-1.10324875e+00,-6.51177768e-01,1.32216344e+00,9.29652453e-01
,-7.92415659e-01,-1.19625322e+00,4.18776090e-01,-6.37262511e-01
,-4.12751182e-01]
    BEV_am_peak_percentage= [1.56209441e+00,-2.26387631e-01,-4.19282866e-01,-1.33250481e+00
,-6.11503740e-01,1.83062600e+00,-2.49819178e-01,-3.86758266e-01
,-1.35152925e-01,-8.01422861e-02,1.10763790e-01,-1.39838054e+00
,-2.79757091e-01,-1.21672571e+00,-4.41333129e-01,-4.83515932e-01
,6.47223976e-01,1.24877458e-01,-3.22718537e-01,-6.24734529e-01
,1.95114485e+00,1.66594861e+00,-7.99052751e-01,1.94840213e+00
,-8.33311305e-01]
    BEV_pm_peak_percentage=[1.50902327e+00,-6.94023762e-01,-2.54840166e-01,5.67481455e-01
,-7.11127384e-01,-2.26585767e-01,-3.76496550e-01,-7.47975956e-01
,-1.25423353e-01,-6.30935879e-01,5.90973782e-01,9.98502590e-01
,-5.29198402e-02,4.76224188e-01,-2.18872051e-01,-2.16573224e+00
,3.02263648e+00,-2.33829599e-01,-7.06576115e-01,-8.00442115e-02
,8.07929658e-01,1.15191573e+00,-2.38681054e-01,-2.15554574e-01
,-1.44506865e+00]
    BEV_weekends_distance_percentage= [1.55202111e-01,-9.51343641e-01,1.60237007e+00,-6.84003091e-02
,-1.25628785e-01,-2.30603038e+00,1.76283564e+00,1.42114768e+00
,7.02634412e-01,-9.23011997e-01,7.44169302e-01,-8.70593717e-01
,2.35202053e-03,-1.05366128e+00,4.18955838e-01,-9.35635720e-01
,-6.74031732e-01,-1.57720478e-01,1.46979387e+00,6.94523562e-01
,-7.72554133e-01,4.72091696e-01,3.56932411e-01,-1.40015079e+00
,4.35754341e-01]
    BEV_charging_rate_mean=[-1.00775227e+00,-7.56950533e-01,-3.62597845e-01,1.31137738e+00
,7.04633601e-01,-7.02342073e-01,-9.14806203e-01,-7.97729157e-01
,2.34761719e+00,-9.79710939e-01,-9.90983866e-01,-3.67490926e-01
,9.24492064e-01,1.65814429e+00,7.95534215e-01,1.27399335e+00
,-7.20069312e-01,-7.62252063e-01,-8.61904527e-01,9.30048972e-01
,-1.02203772e+00,7.66278662e-01,7.34104745e-01,-7.60062400e-01
,-4.39534628e-01]
    BEV_chains_num=[-1.27261757e+00,-3.55873724e-01,2.30253171e-01,-3.02621297e-01
,7.71132320e-01,-1.24149953e+00,3.64045641e-01,-1.91504527e+00
,9.64589484e-01,-1.11262297e+00,-1.24801396e+00,4.00464977e-02
,4.53664888e-01,3.84247412e-01,7.48131752e-01,1.28342453e+00
,-2.30509883e+00,5.05552698e-01,6.65600449e-01,1.49715289e+00
,1.14131570e-01,5.49690090e-01,1.24515224e+00,-6.49898200e-01
,5.86475718e-01]
    BEV_chains_mile_std= [-3.30812936e-01,-2.68602887e-01,-2.98512353e-01,-1.11023339e-01
,-1.79594645e-01,-2.11033937e-01,-2.58966958e-01,-3.81251647e-01
,1.92968343e-01,-3.29791332e-01,-2.18978506e-01,-3.40293264e-02
,-2.22989675e-01,6.16346170e-02,-1.96321155e-01,-1.05363341e-01
,4.86017930e+00,-2.11345754e-01,-2.91584659e-01,-1.40714788e-01
,-2.25438649e-01,-2.97200112e-01,-2.25128585e-01,-3.26049654e-01
,-2.50048022e-01]
    BEV_daily_num_run_status= [-9.78906481e-01,-1.12917846e-01,-9.20045368e-01,7.93286202e-01
,8.08756844e-01,-1.13313074e+00,-7.58008220e-01,-1.19837453e+00
,6.12000142e-01,-8.78363336e-01,-1.09078947e+00,2.42162670e-01
,1.40588053e+00,5.34432337e-01,1.82441731e+00,5.79195295e-01
,-1.16639048e+00,-7.08340923e-01,-7.09025437e-01,2.30080132e+00
,-5.53001216e-01,3.70889018e-03,1.01022074e+00,-7.58327657e-01
,8.50759420e-01]

    '------------------------------------------'
    PHEV_daily_distance_mean=[1.29579442e+00,-1.36821334e+00,8.63435515e-01,8.71928232e-02
,-1.60209012e-01,1.76294617e-01,-2.15171438e-01,-1.49123909e+00
,-1.40022247e+00,4.69502859e-01,-5.59343604e-01,1.64994073e+00
,-2.10547892e-01,4.03847416e-01,1.19307112e+00,-4.51763494e-01
,-7.38977551e-03,1.53377396e+00,7.79682738e-01,1.10069041e+00
,-1.47787736e+00,-1.36701559e+00,-1.51685486e+00,8.68257666e-01
,-1.95636350e-01]
    PHEV_daily_distance_std=[3.81179661e-01,-1.33985652e+00,6.16429675e-01,6.91618359e-01
,-9.57157227e-02,8.35791733e-01,-9.22829789e-01,-1.33388686e+00
,-1.37878194e+00,2.40044165e+00,-1.87347564e-03,7.15357143e-01
,1.12139476e+00,1.61089373e-01,1.32361224e+00,1.05372089e-01
,-5.56717855e-01,-7.73825265e-01,1.76632526e+00,2.41603712e-01
,-1.26246553e+00,-9.74141301e-01,-7.02670369e-01,-5.12041954e-01
,-5.05409085e-01]
    PHEV_night_distance_mean=[9.31787539e-01,-7.95307699e-01,2.68373307e+00,-7.95307699e-01
,-1.41226161e-01,3.18038457e-03,-7.23185125e-01,-7.89849774e-01
,-5.79085679e-01,-1.18620812e-01,-4.51042612e-01,1.58155437e+00
,-7.09442083e-01,8.05484116e-01,1.66166799e+00,4.81626116e-01
,-7.95307699e-01,-7.88221972e-01,-7.95307699e-01,1.15076893e+00
,-7.95307699e-01,-7.86128462e-01,1.29143877e+00,-7.95307699e-01
,-7.32592417e-01]
    PHEV_night_distance_percentage=[1.18739775e+00,-6.57044727e-01,3.47949919e+00,-6.57044727e-01
,2.87372726e-01,-1.08500597e-01,-6.01126850e-01,-6.37392547e-01
,-1.12625824e-01,-1.45968003e-01,-2.63367749e-01,1.60660756e+00
,-4.98126549e-01,7.33768461e-01,-4.73586798e-01,-1.08368276e-02
,-6.57044727e-01,-6.48841699e-01,-6.57044727e-01,1.74544431e+00
,-6.57044727e-01,-6.28670031e-01,-4.43191870e-01,-6.57044727e-01
,-5.25586291e-01]
    PHEV_am_peak_percentage=[-1.12691549e+00,5.23083470e-01,-8.52726545e-01,-4.50571681e-01
,-6.04073603e-01,-5.61731439e-01,2.15171671e+00,1.69421976e+00
,-9.04301954e-02,-1.24165026e+00,1.54022357e+00,-1.47203031e-01
,-1.59105755e-01,-1.55811100e+00,7.75950409e-01,2.72862672e-01
,-3.60079877e-01,3.36678413e-01,-1.33806533e+00,4.05293699e-01
,-1.39219648e+00,-1.46770856e-02,1.02143622e+00,1.42503808e+00
,-2.48965237e-01]
    PHEV_pm_peak_percentage=[2.84187638e-01,-2.22410976e+00,3.01879307e-02,-1.89018590e-01
,1.78001880e-01,3.04339171e-01,2.11860320e+00,2.02827163e+00
,-1.01039805e+00,3.30222872e-01,-1.64618587e-01,-1.02692773e+00
,-1.80166190e-01,-5.58259713e-01,-9.37907629e-02,-3.93054112e-01
,6.48971941e-01,4.63651217e-01,-6.88971360e-01,-1.11278062e+00
,-1.82739518e+00,1.28973448e+00,6.36149498e-01,7.57657689e-01
,3.99511514e-01]
    PHEV_weekends_distance_percentage=[-7.85237644e-01,-5.63822825e-01,-1.09720612e-01,-1.22657011e-01
,-2.98293326e-01,-1.86722689e-01,3.63657817e-01,6.59311270e-01
,-1.73169297e-01,-1.80956330e-01,3.86478328e-01,-1.79093173e-01
,-1.18450929e-01,2.16819292e+00,-1.30507109e+00,2.27120837e-01
,-4.68181428e-01,2.26587690e-01,-1.33814958e+00,3.83146744e-02
,-1.00323623e+00,3.62990061e+00,-5.31611206e-01,-3.24892888e-01
,-1.02978971e-02]
    PHEV_charging_rate_mean=[-1.40398010e+00,-4.92076970e-01,-7.49715421e-02,-2.49540512e-02
,-1.19356947e+00,-4.77593579e-01,-4.77752344e-01,-1.40058052e-01
,-2.67973297e-01,-1.66250845e-01,-4.44420443e-01,7.35263221e-01
,2.76007883e-01,4.44417728e+00,-1.90463921e-01,2.36128002e-01
,-7.83316852e-02,4.14191057e-03,-1.08987865e-01,-4.99955084e-03
,2.15116265e-02,4.78754257e-02,-5.89878821e-02,1.27913888e-01
,-2.87647638e-01]
    PHEV_chains_num=[-1.00662408e+00,2.40893897e-01,-4.50465232e-01,6.20803769e-02
,3.46064921e+00,-9.88195539e-02,-5.38804836e-01,-9.25987151e-01
,2.61725925e-01,-7.36130497e-01,9.09050378e-01,7.69242166e-01
,9.07313973e-01,-1.60558672e+00,3.02871482e-01,1.19392391e+00
,-4.69498282e-01,-8.91623698e-01,-3.07042535e-01,-1.58553830e-02
,-1.06192931e+00,5.15022613e-01,-1.79684337e-01,-8.92026798e-01
,5.57304481e-01]
    PHEV_chains_mile_std=[1.65572340e+00,-1.29801111e+00,1.12506087e+00,3.98393326e-02
,-7.91546991e-01,2.66135094e-01,-4.60473100e-01,-1.25860118e+00
,-1.29619075e+00,2.48740187e+00,-3.76343409e-01,3.36196133e-01
,3.90944166e-01,9.93029730e-01,6.65727284e-01,-3.87628655e-01
,-4.55138678e-01,3.90727791e-01,1.58074506e+00,3.50500296e-01
,-1.14312436e+00,-1.18654510e+00,-8.58387576e-01,-4.64800154e-02
,-7.23560102e-01]
    PHEV_daily_num_run_status=[5.35814528e-01,-7.68410508e-01,2.17595533e-01,-2.32475248e-01
,-2.93359373e-01,-3.53595328e-01,4.61878128e-01,-7.59579533e-01
,-7.46399550e-01,-3.57465992e-01,-4.79006688e-01,3.66493561e+00
,-4.42685219e-01,2.22268043e+00,-1.47064172e-01,-4.89264633e-01
,-9.58301581e-03,9.69577500e-01,-4.04701729e-01,7.91301352e-02
,-7.84837835e-01,-7.56330285e-01,-8.22721179e-01,2.04971509e-01
,-5.09103087e-01]

    # features = np.array([BEV_daily_distance_mean, \
    #               BEV_daily_distance_std, \
    #               BEV_night_distance_mean, \
    #               BEV_night_distance_percentage, \
    #               BEV_am_peak_percentage, \
    #               BEV_pm_peak_percentage, \
    #               BEV_weekends_distance_percentage, \
    #               BEV_charging_rate_mean,
    #               BEV_chains_num, \
    #               BEV_chains_mile_std, \
    #               BEV_daily_num_run_status,\
    #               ])
    # features_label = {0: BEV_daily_distance_mean,
    #                   1: BEV_daily_distance_std,
    #                   2: BEV_night_distance_mean,
    #                   3: BEV_night_distance_percentage,
    #                   4: BEV_am_peak_percentage,
    #                   5: BEV_pm_peak_percentage,
    #                   6: BEV_weekends_distance_percentage,
    #                   7: BEV_charging_rate_mean,
    #                   8: BEV_chains_num,
    #                   9: BEV_chains_mile_std,
    #                  10: BEV_daily_num_run_status,
    #                  # 10: BEV_daily_num_run_status
    #                   }

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

    # 计算每一列的平均值
    # meandata = np.mean(features, axis=0)
    # 均值归一化
    # features = features - meandata
    # 求协方差矩阵
    cov = np.cov(features.T.transpose())
    # 求解特征值和特征向量
    eigVals, eigVectors = np.linalg.eig(cov)
    # 选择前两个特征向量
    pca_mat = eigVectors[:, :3]
    pca_data = np.dot(features.T, pca_mat)
    pca_data = pd.DataFrame(pca_data, columns=['pca1', 'pca2', 'pca3'])

    # 3个主成分的散点图
    plt.subplot(111)
    plt.scatter(pca_data['pca1'], pca_data['pca2'], pca_data['pca3'])
    plt.xlabel('pca_1')
    plt.ylabel('pca_2')
    plt.ylabel('pca_3')
    plt.show()
    # print('前两个主成分包含的信息百分比：{:.2%}'.format(np.sum(eigVals[:2]) / np.sum(eigVals)))
    print('前3个主成分包含的信息百分比：{}'.format(np.sum(eigVals[:3]) / np.sum(eigVals)))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    # for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
        # xs = randrange(n, 23, 32)
        # ys = randrange(n, 0, 100)
        # zs = randrange(n, zlow, zhigh)
        # ax.scatter(xs, ys, zs, c=c, marker=m)
        # ax.scatter(pca_data['pca1'], pca_data['pca2'], pca_data['pca3'], label='parametric curve')
        # ax.scatter(pca_data['pca1'], pca_data['pca2'], pca_data['pca3'], c=c,marker=m)
    ax.scatter(pca_data['pca1'], pca_data['pca2'], pca_data['pca3'])
    plt.show()

    feature_label = [features_label[i] for i in range(len(features))]
    feature_label=np.array(feature_label)
    acc_0,si_score,accuracy=cal(feature_label.T,features_label,[],0,accuracy,si_scores)
    print('----------------------------------------')
    print('The best accuracy: {}'.format(findBestAccuracy(accuracy)))
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

