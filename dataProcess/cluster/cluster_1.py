import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns

from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics

# -*- coding: utf-8 -*-
"""
Created on 2020.8.21
@author: Hou Jue
"""

print('------------------read data--------------------')
'''private car'''
# data_PHEV_private=pd.read_excel('...\\....xlsx',sheet_name='...')
# data_EV_private=pd.read_excel('...\\....xlsx',sheet_name='...')

'''ride_hailing car'''
# data_PHEV_ride_hailing=pd.read_excel('...\\....xlsx',sheet_name='...')

'''taxi'''
# data_EV_taxi=pd.read_excel('...\\....xlsx',sheet_name='...')

'''test'''
data_PHEV_private=pd.read_excel('F:\\PycharmProjects\\ACM\\(Data)Vehicle trajectory\\data\\data\\PHEV_2.xlsx',sheet_name='Raw Data')

''' time
    vehiclestatus
    sumvoltage
    summileage
    sumcurrent
    speed
    soc
    runmodel (1:EV 2:PHEV 3:diesel)
    insulationresisitance
    gearnum
    dcdcstatus
    chargestatus
    ------------------
    longitude
    lon
    locationstate
    latitude
    lat
'''



'''--------------I 数据清洗-------------'''
'''1.重复记录'''






'''2.不完整记录'''
'''数据缺失判断--------------------->begin'''
print('数据缺失判断:',data_PHEV_private.isnull().sum())
print('数据行数',data_PHEV_private.shape[0])
'''数据缺失判断--------------------->end'''

'''数据缺失剔除--------------------->begin'''
data_PHEV_private=data_PHEV_private.dropna()
print('处理后，数据缺失判断:',data_PHEV_private.isnull().sum())
print('处理后，数据行数',data_PHEV_private.shape[0])
'''数据缺失剔除--------------------->end'''


'''3.非当日记录'''


'''4.漂移记录'''


'''--------------II 统计分析--------------'''

data_PHEV_private.info() #查看数据基本信息
print('statistics information',data_PHEV_private.describe()) #查看数据基本信息


fig=plt.figure(figsize=(4,3))

#绘制富人bmi数据直方图
p1=fig.add_subplot(211)
plt.hist(data_PHEV_private['speed'],bins=50,rwidth=0.9)
plt.xlabel('speed')
plt.xlim((0,120))
plt.ylabel('Counts')
plt.title('speed')

'''车辆状态 vehiclestatus'''
p2=fig.add_subplot(212)
plt.hist(data_PHEV_private['summileage'],bins=50,rwidth=0.9)
plt.xlabel('summileage')
# plt.xlim((0,120))
plt.ylabel('Counts')
plt.title('speed')

plt.show()


#
'''data各列数据的统计特征'''
print(data_PHEV_private.describe())
'''data某列类型'''
print(data_PHEV_private.head())
'''核密度估计'''
data_PHEV_private['speed'].plot(kind='kde')
print("max(data['speed']",max(data_PHEV_private['speed']))
print("max(data['speed']",min(data_PHEV_private['speed']))


'''绘制特征两两之间的pearson相关系数矩阵'''
print(data_PHEV_private.corr())

sns.heatmap(data_PHEV_private.corr(),annot=True)
'''箱型图'''
data_PHEV_private.plot(kind='box')
# plt.show()

'''pandas hist'''
data_PHEV_private.hist(bins=20)
# plt.show()

'''------------'''
coords=np.array([data_PHEV_private['latitude'],data_PHEV_private['longitude']])
coords=coords.T

# earth's radius in km
kms_per_radian = 6371.0088
# define epsilon as 0.5 kilometers, converted to radians for use by haversine
''':parameters 0.01 10'''
epsilon = 0.01 / kms_per_radian

# eps is the max distance that points can be from each other to be considered in a cluster
# min_samples is the minimum cluster size (everything else is classified as noise)
db = DBSCAN(eps=epsilon, min_samples=10, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
cluster_labels = db.labels_
# get the number of clusters (ignore noisy samples which are given the label -1)
num_clusters = len(set(cluster_labels) - set([-1]))

# print( 'Clustered ' + str(len(df_min)) + ' points to ' + str(num_clusters) + ' clusters')
print( 'Clustered ' + str(len(coords)) + ' points to ' + str(num_clusters) + ' clusters')
# turn the clusters in to a pandas series
clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
print(clusters)


print('------------------genarete cluster figure-------------------')
from shapely.geometry import MultiPoint
from geopy.distance import great_circle
def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)

# get the centroid point for each cluster
centermost_points = clusters.map(get_centermost_point)
lats, lons = zip(*centermost_points)
rep_points = pd.DataFrame({'lon':lons, 'lat':lats})
fig, ax = plt.subplots(figsize=[10, 6])
rs_scatter = ax.scatter(rep_points['lon'][0], rep_points['lat'][0], c='#dd8a81', edgecolor='None', alpha=0.7, s=450)
ax.scatter(rep_points['lon'][1], rep_points['lat'][1], c='#dd8a81', edgecolor='None', alpha=0.7, s=250)
ax.scatter(rep_points['lon'][2], rep_points['lat'][2], c='#dd8a81', edgecolor='None', alpha=0.7, s=250)
ax.scatter(rep_points['lon'][3], rep_points['lat'][3], c='#dd8a81', edgecolor='None', alpha=0.7, s=150)

'''经纬度相反'''
# for i in range(5):
#     df_scatter = ax.scatter(data_lon_day6, data_lat_day6, c='k', alpha=0.9, s=3)

df_scatter = ax.scatter(data_PHEV_private['longitude'], data_PHEV_private['latitude'],c='k', alpha=0.9, s=3)
ax.set_title('(PHEV Day02) 2016/12/2-2016/12/3 GPS trace and cluster points')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend([df_scatter, rs_scatter], ['GPS points', 'Cluster centers'], loc='upper right')

labels = ['cluster{0}'.format(i) for i in range(1, num_clusters+1)]
for label, x, y in zip(labels, rep_points['lon'], rep_points['lat']):
    plt.annotate(
        label,
        xy = (x, y), xytext = (-25, -30),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=1', fc = 'white', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

plt.show()
print('------------------plt1 shows-------------------')
'''To infer home and work locations, here I used a very simple heuristic: time. Below 
I plot the time distribution of GPS data points in each of the four clusters. We can see that from 9am to 18pm, 
the user stays in the cluster 1 area, while during midnight to 8am, the user tends to stay in cluster 2 and cluster 3. 
Therefore I deduce that user 001’s work location is in cluster 1 and home location is in cluster 2. Cluster 3 might be her secondary residence location.
Of course, we can apply more sophisticated heuristics to infer home and work locations. For example, we can also check users’ 
locations during weekdays and weekends to give us additional clues.'''



''' Get the hours for each cluster  获取每个群集的小时数'''






'''III 特征提取'''


'''Python「第一节」-制作自己的pip安装包
https://blog.csdn.net/ligaopan/article/details/103187590'''





