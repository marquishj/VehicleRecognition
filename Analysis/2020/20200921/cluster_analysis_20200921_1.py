# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 21:40:50 2019

@author: MARs
"""

from sklearn.cluster import DBSCAN
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
#
# user = '001'
# userdata = 'G:\\Geolife Trajectories 1.3\\Data\\' + user + '/Trajectory/'


print('------------------read data--------------------')
import folium
import webbrowser
import pandas as pd

# m = folium.Map(location=[121.1940586, 31.46265569])
# m = folium.Map(location=[31.46265569, 121.1940586])
# m.save("1.html")
# webbrowser.open("1.html")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import datatime


data_PHEV_private=pd.read_excel('F:\\PycharmProjects\\ACM\\(Data)Vehicle trajectory\data\\PHEV样本数据.xlsx',sheet_name='Raw Data')


data_PHEV_private['数据采集时间']=pd.to_datetime(data_PHEV_private['数据采集时间'])
'''数据采集时间  作为索引'''
data=data_PHEV_private.set_index('数据采集时间')
data_lat = data['经度']
data_lon = data['纬度']
data_coords=np.array([data_lon,data_lat])
data_coords=np.transpose(data_coords)

'''数据缺失判断--------------------->begin'''

print('数据缺失判断:data_lat',data_lat.isnull().sum())
print('数据缺失判断:data_lon',data_lon.isnull().sum())


'''数据缺失判断--------------------->end'''

'''按每天的轨迹作图'''
'''3、时间数据的处理'''
data_PHEV_private['数据采集时间']=pd.to_datetime(data_PHEV_private['数据采集时间'])
data_PHEV_private=data_PHEV_private.set_index('数据采集时间')
print(np.size(data_PHEV_private))
print(data_PHEV_private['2016/12/1 00:00:00':'2016/12/2 0:00:00'])
''''''
print('-------------------------------------------------')
print(data_PHEV_private['2016/12/1 00:00:00':'2016/12/2 23:59:59']['纬度'],data_PHEV_private['2016/12/1 00:00:00':'2016/12/2 23:59:59']['经度'])
print(data_PHEV_private.head())
print('------------------read data done!--------------------')

data_lat=data_PHEV_private['纬度']
data_lon=data_PHEV_private['经度']

data_lat_day1=data_PHEV_private['2016/12/1 00:00:00':'2016/12/1 23:59:59']['纬度']
data_lon_day1=data_PHEV_private['2016/12/1 00:00:00':'2016/12/1 23:59:59']['经度']

data_lat_day2=data_PHEV_private['2016/12/2 00:00:00':'2016/12/2 23:59:59']['纬度']
data_lon_day2=data_PHEV_private['2016/12/2 00:00:00':'2016/12/2 23:59:59']['经度']

data_lat_day3=data_PHEV_private['2016/12/3 00:00:00':'2016/12/3 23:59:59']['纬度']
data_lon_day3=data_PHEV_private['2016/12/3 00:00:00':'2016/12/3 23:59:59']['经度']
#
data_lat_day4=data_PHEV_private['2016/12/4 00:00:00':'2016/12/4 23:59:59']['纬度']
data_lon_day4=data_PHEV_private['2016/12/4 00:00:00':'2016/12/4 23:59:59']['经度']

data_lat_day5=data_PHEV_private['2016/12/5 00:00:00':'2016/12/5 23:59:59']['纬度']
data_lon_day5=data_PHEV_private['2016/12/5 00:00:00':'2016/12/5 23:59:59']['经度']

data_lat_day6=data_PHEV_private['2016/12/6 00:00:00':'2016/12/6 23:59:59']['纬度']
data_lon_day6=data_PHEV_private['2016/12/6 00:00:00':'2016/12/6 23:59:59']['经度']

data_lat_day7=data_PHEV_private['2016/12/7 00:00:00':'2016/12/7 23:59:59']['纬度']
data_lon_day7=data_PHEV_private['2016/12/7 00:00:00':'2016/12/7 23:59:59']['经度']

data_lat_day8=data_PHEV_private['2016/12/8 00:00:00':'2016/12/8 23:59:59']['纬度']
data_lon_day8=data_PHEV_private['2016/12/8 00:00:00':'2016/12/8 23:59:59']['经度']

data_lat_day9=data_PHEV_private['2016/12/9 00:00:00':'2016/12/9 23:59:59']['纬度']
data_lon_day9=data_PHEV_private['2016/12/9 00:00:00':'2016/12/9 23:59:59']['经度']

data_lat_day10=data_PHEV_private['2016/12/10 00:00:00':'2016/12/10 23:59:59']['纬度']
data_lon_day10=data_PHEV_private['2016/12/10 00:00:00':'2016/12/10 23:59:59']['经度']

'''data datetime choice---->begin'''
# data_day1=np.array([data_lat_day1,data_lon_day1])
# data_coords1=np.transpose(data_day1)

# data_all_days=np.array([data_lat_day1,data_lon_day1])
data_all_days=np.array([data_lat_day2,data_lon_day2])

coords=np.transpose(data_all_days)
coords=np.round(coords, 6)


# earth's radius in km
kms_per_radian = 6371.0088
# define epsilon as 0.5 kilometers, converted to radians for use by haversine
''':parameters 0.01 10'''
# epsilon = 0.01 / kms_per_radian
epsilon = 0.1 / kms_per_radian

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
