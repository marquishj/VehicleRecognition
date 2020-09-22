import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import sys
import datetime
import math
from sklearn.cluster import DBSCAN
pi = 3.1415926535897932384626  # π
ee = 0.00669342162296594323
a = 6378245.0

# -*- coding: utf-8 -*-
# @Time    : 2020/9/22 14:21
# @Author  : Hou Jue

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

class trajectoryAnalysis():
    def __init__(self):
        pass

    def dataPrePorcess(self,flag,data_byDay):

        if flag==1:
            # pcev_list = os.listdir("D:\\data\\私家车\\纯电\\")
            # pcphev_list= os.listdir("D:\\data\\私家车\\混动\\")
            # didi_list = os.listdir("D:\\data\\网约车\\")
            didi_list = os.listdir("F:\\sql data\\classifer_car_data\\example\\vehicle\\")
            # taxi_list = os.listdir("D:\\data\\出租车\\")

            ''' didi-EV'''
            for didi_id in range(len(didi_list)):
                # data_didi_HEV = pd.read_csv('D:\\data\\网约车\\' + didi_list)
                '''这里已经是一个循环了 didi_list[didi_id]
                   分别取出车辆ID'''
                data_didi_HEV = pd.read_csv('F:\\sql data\\classifer_car_data\\example\\vehicle\\' + didi_list[didi_id])
                data_didi_HEV['datatime'] = pd.to_datetime(data_didi_HEV['datatime']).apply(lambda x: x.date())

        # data_didi_HEV=data_byDay
        data_didi_HEV['lng_new'] = 0.000000
        data_didi_HEV['lat_new'] = 0.000000

        for i in range(data_didi_HEV.index.tolist()[0],data_didi_HEV.index.tolist()[-1]):
            data_didi_HEV['lng_new'][i] = self.transfer(data_didi_HEV['longitude'][i], data_didi_HEV['latitude'][i])[0]
            data_didi_HEV['lat_new'][i] = self.transfer(data_didi_HEV['longitude'][i], data_didi_HEV['latitude'][i])[1]

        '''应该生成excel保存'''
        data_didi_HEV.to_csv()



        return  data_didi_HEV

    def _transformlat(self,lng, lat):
        ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
              0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
        ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
                math.sin(2.0 * lng * pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lat * pi) + 40.0 *
                math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
        ret += (160.0 * math.sin(lat / 12.0 * pi) + 320 *
                math.sin(lat * pi / 30.0)) * 2.0 / 3.0
        return ret

    def _transformlng(self,lng, lat):
        ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
              0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
        ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
                math.sin(2.0 * lng * pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lng * pi) + 40.0 *
                math.sin(lng / 3.0 * pi)) * 2.0 / 3.0
        ret += (150.0 * math.sin(lng / 12.0 * pi) + 300.0 *
                math.sin(lng / 30.0 * pi)) * 2.0 / 3.0
        return ret

    def transfer(self,lng, lat):
        dlat = self._transformlat(lng - 105.0, lat - 35.0)
        dlng = self._transformlng(lng - 105.0, lat - 35.0)
        radlat = lat / 180.0 * pi
        magic = math.sin(radlat)
        magic = 1 - ee * magic * magic
        sqrtmagic = math.sqrt(magic)
        dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
        dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
        mglat = lat + dlat
        mglng = lng + dlng
        return [mglng, mglat]

    def divideDay(self,day):

        '''在这里修改车型'''
        # data_didi_HEV = pd.read_csv('D:\\data\\出租车\\SHEVDC_1A5H2N7C.csv')
        '''didi'''
        # data_didi_HEV = pd.read_csv('D:\\2020.09.10\\HJ\\pick_vehicle\\didi\\SHEVDC_546B660B.csv')
        '''pc_ev'''
        # data_didi_HEV = pd.read_csv('D:\\2020.09.10\\HJ\\pick_vehicle\\pc_ev\\SHEVDC_3M4S302G.csv')
        '''pc_hev'''
        # data_didi_HEV = pd.read_csv('D:\\2020.09.10\\HJ\`\pick_vehicle\\pc_hev\\SHEVDC_4L10316B.csv')
        '''taxi'''
        # data_didi_HEV = pd.read_csv('D:\\2020.09.10\\HJ\\pick_vehicle\\taxi\\SHEVDC_341O5511.csv') #输入车辆名称的文件名
        # data_didi_HEV = pd.read_csv('F:\\sql data\\classifer_car_data\\example\\SHEVDC_0A101F56_vehicle_position.csv') #输入车辆名称的文件名
        flag = 1
        data_byDay=0
        '''感觉这里要循环处理？分别计算文件夹里的表格文件'''
        data_didi_HEV = self.dataPrePorcess(flag,data_byDay)#输入车辆名称的文件名
        data_didi_HEV['datatime'] = pd.to_datetime(data_didi_HEV['datatime']).apply(lambda x: x.date())

        days = str(data_didi_HEV['datatime'])
        if day in days:
            data_byDay_index=data_didi_HEV[data_didi_HEV.loc[:, 'datatime']== pd.to_datetime(day)].index.tolist()
            data_byDay=data_didi_HEV.loc[data_byDay_index,:]
        else:
            data_byDay=pd.DataFrame([np.nan,np.nan],columns=['datatime'])

        return data_byDay,data_didi_HEV

    def clusterAnalysis(self,days):
        print('--------------ClusterAnalysis  Start-----------------')
        for day in days:
            data_byDay, data_didi_HEV=self.divideDay(day)
            if pd.isnull(data_byDay['datatime']).iloc[0] == True:
                print("{}\nCannot find this day from datasets.".format(day))
                print('---------------------------------')
            else:
                lat_byDay,lon_byDay= np.array(data_byDay['latitude']),np.array(data_byDay['longitude'])
                coords=np.round(np.transpose(np.array([lat_byDay,lon_byDay])),6)
                kms_per_radian = 6371.0088
                ''':parameters 0.01 10'''
                epsilon = 0.2 / kms_per_radian
                db = DBSCAN(eps=epsilon, min_samples=10, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
                cluster_labels = db.labels_
                num_clusters = len(set(cluster_labels) - set([-1]))
                '''找num_clusters对应的时间范围'''
                # for i in num_clusters:
                #     data_byDay['datatime'] cluster_labels[cluster_labels==i].index

                print('{}'.format(day))
                print('Clustered ' + str(len(coords)) + ' points to ' + str(num_clusters) + ' clusters')
                print('---------------------------------')

        print('---------------ClusterAnalysis  Finish-----------------')

    def POIPorcess(self):

        # df_airport=pd.read_excel('C:\\Users\\admin\\PycharmProjects\\pythonProject\\20200910\\match_POI\\data\\airport.xlsx')
        df_airport=pd.read_excel('airport.xlsx')
        # df_railway=pd.read_excel('C:\\Users\\admin\\PycharmProjects\\pythonProject\\20200910\\match_POI\\data\\railway_station.xlsx')
        df_railway=pd.read_excel('railway_station.xlsx')

        '''airport'''
        df_airport_location=df_airport.loc[:,'location']
        airport_location=list(df_airport_location)
        pd_airport_location=pd.DataFrame(df_airport_location)
        pd_airport_location.loc[0,'location'].split(',')
        '''railway station'''
        df_railway_location = df_railway.loc[:, 'location']
        railway_location = list(df_railway_location)
        pd_railway_location = pd.DataFrame(df_railway_location)
        pd_railway_location.loc[0, 'location'].split(',')

        airport_location_volume=pd_airport_location.shape[0]
        railway_location_volume=pd_railway_location.shape[0]
        '''airport'''
        airport_location_2_column=[pd_airport_location.loc[i,'location'].split(',') for i in range(airport_location_volume)]
        '''railway station'''
        railway_location_2_column = [pd_railway_location.loc[i, 'location'].split(',') for i in range(railway_location_volume)]
        airport_location_done=list()
        '''airport'''
        for i in range(airport_location_volume):
            airport_location_done.append(list(map(eval, airport_location_2_column[i])))
        '''二维数组再转为DataFrame格式，可以了'''
        airport_location_done=pd.DataFrame(airport_location_done,columns=['longitude','latitude'])
        '''railway station'''
        railway_location_done = list()
        for i in range(railway_location_volume):
            railway_location_done.append(list(map(eval, railway_location_2_column[i])))
        railway_location_done = pd.DataFrame(railway_location_done, columns=['longitude', 'latitude'])
        s1 = ','.join(str(n) for n in airport_location)

        return airport_location_done,railway_location_done

    def matchPOI(self,data_byDay):

        airport_location, railway_location=self.POIPorcess()
        flag=0
        data_didi_HEV = self.dataPrePorcess(flag,data_byDay)
        data_didi_HEV_location=pd.concat([data_didi_HEV['lat_new'],data_didi_HEV['lng_new']],axis=1)

        '''airport'''
        match_points_airport=0
        for i in range(airport_location.shape[0]):
            for j in range(data_didi_HEV_location.index[0],data_didi_HEV_location.index[-1]):
                if (data_didi_HEV_location.loc[j,'lat_new'].round(decimals=2)==airport_location.loc[i,'latitude'].round(decimals=2))\
                        and (data_didi_HEV_location.loc[j,'lng_new'].round(decimals=2)==airport_location.loc[i,'longitude'].round(decimals=3)):
                    match_points_airport+=1

        '''railway station'''
        match_points_railway = 0
        for i in range(railway_location.shape[0]):
            for j in range(data_didi_HEV_location.index[0],data_didi_HEV_location.index[-1]):
                if (data_didi_HEV_location.loc[j,'lat_new'].round(decimals=2)==railway_location.loc[i,'latitude'].round(decimals=2))\
                        and (data_didi_HEV_location.loc[j,'lng_new'].round(decimals=2)==railway_location.loc[i,'longitude'].round(decimals=2)):
                    match_points_railway+=1

        return match_points_airport,match_points_railway

    def POIAnalysis(self,days):
        starttime = datetime.datetime.now()
        print('--------------POIAnalysis  Start-----------------')
        for day in days:
            data_byDay,data_didi_HEV = self.divideDay(day)
            if pd.isnull(data_byDay['datatime']).iloc[0] == True:
                print("{}\nCannot find this day from datasets.".format(day))
                print('---------------------------------')
            else:
                match_points_airport, match_points_railway = self.matchPOI(data_byDay)
                print('{}'.format(day))
                print('match_points_airport:{}\nmatch_points_railway_station:{}'.format(match_points_airport,
                                                                                        match_points_railway))
                endtime = datetime.datetime.now()
                print('CPU time(s):', (endtime - starttime).seconds)
                print('---------------------------------')
        print('--------------POIAnalysis  Finish-----------------')

if __name__ == '__main__':

    days = ['2018-01-06', '2018-01-07','2018-01-08']
    traAnalysis = trajectoryAnalysis()
    '''根据分析需要选择功能'''
    '''1-处理轨迹匹配POI'''
    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    sys.stdout = Logger('F:\\sql data\\classifer_car_data\\log.txt')

    traAnalysis.POIAnalysis(days)

    '''2-处理轨迹聚类'''
    traAnalysis.clusterAnalysis(days)
