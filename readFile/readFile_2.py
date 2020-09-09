# ----Hou Jue  2020.08.21----

import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
from sklearn import tree
import matplotlib.pyplot as plt

# -*- coding: utf-8 -*-
"""
Created on 2020.8.21

@author: MARs
"""

import os
#

print('------------------read data--------------------')
import folium
import webbrowser

import csv

# import datatime
'''
车机号（车辆唯一标识）
控制字（A：正常）
业务状态（0：正常，1：报警）
载客状态（0：重车，1：空车）
顶灯状态（0：营运，1：待运，2：电调，3：暂停，4：求助，5：停运）
业务状态（0：地面道路，1：快速道路）
业务状态（0：无刹车，1：刹车）
无意义字段
数据接收时间
终端GPS时间
经度
纬度
速度
方向
卫星数
无意义字段
'''

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
    
    longitude
    lon
    locationstate
    latitude
    lat
'''
# data_taxi=pd.read_csv('F:\\PycharmProjects\\ACM\\(Data)Vehicle trajectory\\data\\taxi.csv')
# data=pd.read_csv(f,names=['数据采集时间',' ','累积行驶里程','定位状态','东经.西经',\
#                         '北纬.南纬','经度','维度','方向',\
#                           '速度','电机控制器温度','驱动电机转速','驱动电机温度','电机母线电流','加速踏板行程','制动踏板状态',\
#                           '动力系统就绪','电池剩余电量(SOC)','电池剩余能量','高压电池电流','电池总电压','单体最高温', \
#                           '单体最低温度','单体最高电压','单体最低电压','绝缘电阻值','电池包最高温度','电池包最高温度_1',\
#                           '电池包最低温度','电池包最低温度_1','电池均衡激活',\
#                           '紧急下电请求','启动时间','液体燃料消耗量','上下线状态','熄火时间','车辆当前状态'])
# data=pd.read_csv(f,names=['ID','status_control','status_work','status_passenger','status_light',\
#                         'status_lane','status_break','null','time_receive',\
#                           'time_GPS','lat','lon','speed','direction','satellite','null'],sep='|')
#
'''data各列数据的统计特征'''
print(data_PHEV_private.describe())
'''data某列类型'''
# print("data['定位状态'].unique()",data_PHEV_private['定位状态'].unique())
# print("data.unique()",data.unique())
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
plt.show()
# data_PHEV_private['数据采集时间']=pd.to_datetime(data_PHEV_private['数据采集时间'])

'''pandas hist'''
data_PHEV_private.hist(bins=20)
plt.show()

coords=np.array([data_PHEV_private['latitude'],data_PHEV_private['longitude']])





from sklearn.cluster import DBSCAN
from sklearn import metrics

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
# ax.scatter(rep_points['lon'][4], rep_points['lat'][4], c='#99cc99', edgecolor='None', alpha=0.7, s=150)
# df_scatter = ax.scatter(df_min['lng'], df_min['lat'], c='k', alpha=0.9, s=3)
# df_scatter = ax.scatter(data_lat, data_lon, c='k', alpha=0.9, s=3)
'''经纬度相反'''
# for i in range(5):
#     df_scatter = ax.scatter(data_lon_day6, data_lat_day6, c='k', alpha=0.9, s=3)




df_scatter = ax.scatter(data_lon_day2, data_lat_day2,c='k', alpha=0.9, s=3)
# df_scatter = ax.scatter(data_lon, data_lat,c='k', alpha=0.9, s=3)
# ax.set_title('Full GPS trace vs. DBSCAN clusters')
# ax.set_title('(PHEV Day01) 2016/12/1-2016/12/2 GPS trace and cluster points')
# ax.set_title('(PHEV Day01) 2016/12/1-2016/12/2 GPS trace and cluster points')
ax.set_title('(PHEV Day02) 2016/12/2-2016/12/3 GPS trace and cluster points')
# ax.set_title('(PHEV Day03) 2016/12/3-2016/12/4 GPS trace and cluster points')
# ax.set_title('(PHEV Day04) 2016/12/4-2016/12/5 GPS trace and cluster points')
# ax.set_title('(PHEV Day05) 2016/12/5-2016/12/6 GPS trace and cluster points')
# df_scatter = ax.scatter(data_lon_day6, data_lat_day6,c='k', alpha=0.9, s=3)
# ax.set_title('(PHEV Day06) 2016/12/6-2016/12/7 GPS trace and cluster points')
#
# df_scatter = ax.scatter(data_lon_day7, data_lat_day7,c='k', alpha=0.9, s=3)
# ax.set_title('(PHEV Day07) 2016/12/7-2016/12/8 GPS trace and cluster points')
# ax.set_title('(PHEV Day08) 2016/12/8-2016/12/9 GPS trace and cluster points')
# ax.set_title('(PHEV Day09) 2016/12/9-2016/12/10 GPS trace and cluster points')
# ax.set_title('(PHEV Day10) 2016/12/10-2016/12/11 GPS trace and cluster points')
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














'''数据缺失判断--------------------->begin'''


# print('数据缺失判断:data_lat',data_lat.isnull().sum())
# print('数据缺失判断:data_lon',data_lon.isnull().sum())



'''数据缺失判断--------------------->end'''



'''查看是否有缺失值'''


'''I 数据清洗'''
'''1.重复记录'''


'''2.不完整记录'''


'''3.非当日记录'''


'''4.漂移记录'''


'''II 统计分析'''


'''III 特征提取'''


'''Python「第一节」-制作自己的pip安装包
https://blog.csdn.net/ligaopan/article/details/103187590'''





