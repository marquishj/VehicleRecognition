# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 21:40:50 2019

@author: MARs
"""

import gmplot
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
# f=open('F:\\Pro xie.chi\\任务：carsharing\\新能源车辆数据中心\\BEV样本数据.xlsx',sheet_name='Raw Data')
# data_BEV_private=pd.read_excel('F:\\Pro xie.chi\\任务：carsharing\\新能源车辆数据中心\\BEV样本数据.xlsx',sheet_name='Raw Data')
# data_taxi=pd.read_excel('F:\\PycharmProjects\\ACM\\(Data)Vehicle trajectory\data\\PHEV样本数据.xlsx',sheet_name='Raw Data')
# data_taxi=pd.read_csv('F:\\Pro xie.chi\\任务：Carsharing\\新能源车辆数据中心\\Taxi_105.txt')
data_taxi=pd.read_excel('F:\\Marquez\\任务：Carsharing\\新能源车辆数据中心\\BEV样本数据.xlsx',sheet_name='Raw Data')

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
# data=[]
# data_lat=data_taxi['纬度']
# data_lon=data_taxi['经度']
#
# data_lat=data_lat.dropna()
# data_lon=data_lon.dropna()





# data_taxi['数据采集时间']
data_taxi=data_taxi.set_index('数据采集时间')
# print(np.size(data_taxi))
# print(data_taxi['2016/12/1 00:00:00':'2016/12/2 0:00:00'])
# ''''''
# print('-------------------------------------------------')
# print(data_taxi['2016/12/1 00:00:00':'2016/12/2 23:59:59']['纬度'],data_taxi['2016/12/1 00:00:00':'2016/12/2 23:59:59']['经度'])
# print(data_taxi.head())
# print('------------------read data done!--------------------')



# data_lat_day1=data_taxi['2017/1/1 00:00:00':'2017/1/1 23:59:59']['纬度']
# data_lon_day1=data_taxi['2017/1/1 00:00:00':'2017/1/1 23:59:59']['经度']

# data_lat_day1=data_taxi['2017/1/2 00:00:00':'2017/1/2 23:59:59']['纬度']
# data_lon_day1=data_taxi['2017/1/2 00:00:00':'2017/1/2 23:59:59']['经度']

# data_lat_day1=data_taxi['2017/1/3 00:00:00':'2017/1/3 23:59:59']['纬度']
# data_lon_day1=data_taxi['2017/1/3 00:00:00':'2017/1/3 23:59:59']['经度']

# data_lat_day1=data_taxi['2017/1/4 00:00:00':'2017/1/4 23:59:59']['纬度']
# data_lon_day1=data_taxi['2017/1/4 00:00:00':'2017/1/4 23:59:59']['经度']

# data_lat_day1=data_taxi['2017/1/5 00:00:00':'2017/1/5 23:59:59']['纬度']
# data_lon_day1=data_taxi['2017/1/5 00:00:00':'2017/1/5 23:59:59']['经度']

data_lat_day1=data_taxi['2017/1/6 00:00:00':'2017/1/6 23:59:59']['纬度']
data_lon_day1=data_taxi['2017/1/6 00:00:00':'2017/1/6 23:59:59']['经度']

# data_lat_day1=data_taxi['2017/1/7 00:00:00':'2017/1/7 23:59:59']['纬度']
# data_lon_day1=data_taxi['2017/1/7 00:00:00':'2017/1/7 23:59:59']['经度']
# #
# data_lat_day1=data_taxi['2017/1/8 00:00:00':'2017/1/8 23:59:59']['纬度']
# data_lon_day1=data_taxi['2017/1/8 00:00:00':'2017/1/8 23:59:59']['经度']
#
# data_lat_day1=data_taxi['2017/1/9 00:00:00':'2017/1/9 23:59:59']['纬度']
# data_lon_day1=data_taxi['2017/1/9 00:00:00':'2017/1/9 23:59:59']['经度']
# int a=1

data_lat_day1=data_lat_day1.dropna()
data_lon_day1=data_lon_day1.dropna()


# data_taxi['lat']=data_taxi.iloc[:,[3]]
# data_taxi['lon']=data_taxi.iloc[:,[2]]
# data_taxi.loc[:,4]

# data_taxi['time']=data_taxi.iloc[:,[1]]
'''数据采集时间  作为索引'''
# # data=data_taxi.set_index('数据采集时间')
# data_lat = data_taxi['lon']
# data_lon = data_taxi['lat']
# data_coords=np.array([data_lon_day1,data_lat_day1])
data_coords=np.array([data_lat_day1,data_lon_day1])
# data_coords=np.array([data_lat_day1,data_lon_day1])
coords=np.transpose(data_coords)
# coords=data_coords








# data_PHEV_private=pd.read_excel('F:\\PycharmProjects\\ACM\\(Data)Vehicle trajectory\data\\PHEV样本数据.xlsx',sheet_name='Raw Data')
#
# # data_taxi=pd.read_csv('F:\\PycharmProjects\\ACM\\(Data)Vehicle trajectory\\data\\taxi.csv')
# # data=pd.read_csv(f,names=['数据采集时间',' ','累积行驶里程','定位状态','东经.西经',\
# #                         '北纬.南纬','经度','维度','方向',\
# #                           '速度','电机控制器温度','驱动电机转速','驱动电机温度','电机母线电流','加速踏板行程','制动踏板状态',\
# #                           '动力系统就绪','电池剩余电量(SOC)','电池剩余能量','高压电池电流','电池总电压','单体最高温', \
# #                           '单体最低温度','单体最高电压','单体最低电压','绝缘电阻值','电池包最高温度','电池包最高温度_1',\
# #                           '电池包最低温度','电池包最低温度_1','电池均衡激活',\
# #                           '紧急下电请求','启动时间','液体燃料消耗量','上下线状态','熄火时间','车辆当前状态'])
# # data=pd.read_csv(f,names=['ID','status_control','status_work','status_passenger','status_light',\
# #                         'status_lane','status_break','null','time_receive',\
# #                           'time_GPS','lat','lon','speed','direction','satellite','null'],sep='|')
# #
#
# data_PHEV_private['数据采集时间']=pd.to_datetime(data_PHEV_private['数据采集时间'])
# '''数据采集时间  作为索引'''
# data=data_PHEV_private.set_index('数据采集时间')
# data_lat = data['经度']
# data_lon = data['纬度']
# data_coords=np.array([data_lon,data_lat])
# data_coords=np.transpose(data_coords)
#
# '''数据缺失判断--------------------->begin'''
#
#
# print('数据缺失判断:data_lat',data_lat.isnull().sum())
# print('数据缺失判断:data_lon',data_lon.isnull().sum())
#
#
#
# '''数据缺失判断--------------------->end'''
#
#
# '''按每天的轨迹作图'''
# '''3、时间数据的处理'''
# data_PHEV_private['数据采集时间']=pd.to_datetime(data_PHEV_private['数据采集时间'])
# data_PHEV_private=data_PHEV_private.set_index('数据采集时间')
#
#
# data_lat=data_PHEV_private['纬度']
# data_lon=data_PHEV_private['经度']
#
# data_lat_day1=data_PHEV_private['2016/12/1 00:00:00':'2016/12/1 23:59:59']['纬度']
# data_lon_day1=data_PHEV_private['2016/12/1 00:00:00':'2016/12/1 23:59:59']['经度']
#
# '''data datetime choice---->begin'''
# # data_day1=np.array([data_lat_day1,data_lon_day1])
# # data_coords1=np.transpose(data_day1)
#
# data_all_days=np.array([data_lat_day1,data_lon_day1])
# # data_coords1=np.transpose(data_all_days)
# coords=np.transpose(data_all_days)
coords=np.round(coords, 8)







'''数据缺失判断--------------------->begin'''

#
# print('数据缺失判断:data_lat',data_lat.isnull().sum())
# print('数据缺失判断:data_lon',data_lon.isnull().sum())




'''day 03'''
# gmap = gmplot.GoogleMapPlotter(data_lat[0], data_lon[0], 11)
# # gmap = gmplot.GoogleMapPlotter(df_min.lat[0], df_min.lng[0], 11)
# # gmap.plot(df_min.lat, df_min.lng)
# gmap.plot(data_lat, data_lon)
# gmap.draw("taxi_map_day01.html")
#
# '''--------------------------------------------------------'''
incidents = folium.map.FeatureGroup()
latitude = 31.23
longitude = 121.47
# #
for lat, lng, in zip(data_lat_day1,
                     data_lon_day1):
    incidents.add_child(
        folium.CircleMarker(
            [lat, lng],
            radius=5, # define how big you want the circle markers to be
            color='yellow',
            fill=True,
            fill_color='red',
            fill_opacity=0.4
        )
    )

# Add incidents to map
san_map = folium.Map(location=[latitude, longitude], zoom_start=12)
san_map.add_child(incidents)
san_map.save("taxi_1.html")
webbrowser.open("taxi_1.html")

'''--------------------------------------DBSCAN--------------------->end'''
from sklearn.cluster import DBSCAN

# earth's radius in km
kms_per_radian = 6371.0088
# define epsilon as 0.5 kilometers, converted to radians for use by haversine
''':parameters 0.01 12'''
epsilon = 10 / kms_per_radian

# eps is the max distance that points can be from each other to be considered in a cluster
# min_samples is the minimum cluster size (everything else is classified as noise)
# db = DBSCAN(eps=epsilon, min_samples=20, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
db = DBSCAN(eps=epsilon, min_samples=20, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
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
rs_scatter = ax.scatter(rep_points['lon'][0], rep_points['lat'][0], c='#dd8a81', edgecolor='None', alpha=0.7, s=250)
# ax.scatter(rep_points['lon'][1], rep_points['lat'][1], c='#dd8a81', edgecolor='None', alpha=0.7, s=250)
# ax.scatter(rep_points['lon'][2], rep_points['lat'][2], c='#dd8a81', edgecolor='None', alpha=0.7, s=250)
# ax.scatter(rep_points['lon'][3], rep_points['lat'][3], c='#dd8a81', edgecolor='None', alpha=0.7, s=250)
# ax.scatter(rep_points['lon'][4], rep_points['lat'][4], c='#dd8a81', edgecolor='None', alpha=0.7, s=250)
# df_scatter = ax.scatter(df_min['lng'], df_min['lat'], c='k', alpha=0.9, s=3)
# df_scatter = ax.scatter(data_lat, data_lon, c='k', alpha=0.9, s=3)
'''经纬度相反'''
# for i in range(5):
#     df_scatter = ax.scatter(data_lon_day6, data_lat_day6, c='k', alpha=0.9, s=3)




df_scatter = ax.scatter(data_lon_day1, data_lat_day1,c='k', alpha=0.9, s=3)
# df_scatter = ax.scatter(data_lon, data_lat,c='k', alpha=0.9, s=3)
# ax.set_title('Full GPS trace vs. DBSCAN clusters')
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
plt.grid()
plt.show()
print('------------------plt1 shows-------------------')
'''To infer home and work locations, here I used a very simple heuristic: time. Below 
I plot the time distribution of GPS data points in each of the four clusters. We can see that from 9am to 18pm, 
the user stays in the cluster 1 area, while during midnight to 8am, the user tends to stay in cluster 2 and cluster 3. 
Therefore I deduce that user 001’s work location is in cluster 1 and home location is in cluster 2. Cluster 3 might be her secondary residence location.
Of course, we can apply more sophisticated heuristics to infer home and work locations. For example, we can also check users’ 
locations during weekdays and weekends to give us additional clues.'''


'''按每天的轨迹作图'''
'''3、时间数据的处理'''
