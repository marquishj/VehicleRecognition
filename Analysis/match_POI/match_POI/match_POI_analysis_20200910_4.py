import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
import datetime
sys.path.append(r"F:\PycharmProjects\VehicleRecognition\Analysis\match_POI\match_POI")
import coord_transfer

'''@Author Hou Jue
   20200910'''


def dataPrePorcess(flag):


    if flag==1:
        pcev_list = os.listdir("D:\\data\\私家车\\纯电\\")
        pcphev_list= os.listdir("D:\\data\\私家车\\混动\\")
        didi_list = os.listdir("D:\\data\\网约车\\")
        taxi_list = os.listdir("D:\\data\\出租车\\")

        ''' didi-EV'''
        for didi_id in didi_list:
            data_didi_HEV = pd.read_csv('D:\\data\\网约车\\' + didi_list)
            # print("missing data" + didi_list, data_didi_HEV.isnull().sum())
            # print("rows in dataset" + didi_list, data_didi_HEV.shape[0])
            # data_didi_HEV = data_didi_HEV.dropna()
            # print("missing data-processed" + didi_list, data_didi_HEV.isnull().sum())
            # print("rows in dataset-processed" + didi_list, data_didi_HEV.shape[0])
            data_didi_HEV['time'] = pd.to_datetime(data_didi_HEV['time']).apply(lambda x: x.date())

    # data_didi_HEV = pd.read_csv('F:\\sql data\\classifer_car_data\\example\\data\\SHEVDC_0A101F56_vehicle_data.csv')
    data_didi_HEV = pd.read_excel('F:\Marquez\任务：Carsharing\新能源车辆数据中心\\BEV样本数据.xlsx')
    data_didi_HEV['lng_new'] = 0.000000
    data_didi_HEV['lat_new'] = 0.000000


    for i in range(0, len(data_didi_HEV)):
        # 84坐标系转高德坐标系
        '''gaode wgs84_to_gcj02'''
        data_didi_HEV['lng_new'][i] = coord_transfer.wgs84_to_gcj02(data_didi_HEV['经度'][i], data_didi_HEV['纬度'][i])[0]
        data_didi_HEV['lat_new'][i] = coord_transfer.wgs84_to_gcj02(data_didi_HEV['经度'][i], data_didi_HEV['纬度'][i])[1]

    return  data_didi_HEV



def divideDay():
    data_didi_HEV = pd.read_excel('F:\Marquez\任务：Carsharing\新能源车辆数据中心\\BEV样本数据.xlsx')
    data_didi_HEV['数据采集时间'] = pd.to_datetime(data_didi_HEV['数据采集时间']).apply(lambda x: x.date())
    a=1
    # agg=data_didi_HEV.groupby('数据采集时间').agg({'数据采集时间'})
    agg_lon=data_didi_HEV.groupby('经度',axis=0).indices
    agg_lat=data_didi_HEV.groupby('纬度',axis=0).indices
    # agg_location=[agg_lon,agg_lat]

    '''to Dataframe'''
    # agg_byDay=pd.DataFrame.from_dict(agg, orient='index')
    # agg_byDay=pd.DataFrame.from_dict(agg)

    # agg_index=agg.index()
# df = pd.read_csv('...\\....csv')

def POIPorcess():



    # data_didi_HEV=pd.read_csv('F:\\sql data\\classifer_car_data\\example\\data\\SHEVDC_0A101F56_vehicle_data.csv')
    # data_didi_HEV['datatime'] = pd.to_datetime(data_didi_HEV['datatime']).apply(lambda x: x.date())
    # data_didi_HEV['datatime'] = pd.to_datetime(data_didi_HEV['datatime']).dt.date



    df_airport=pd.read_excel('F:\\PycharmProjects\\VehicleRecognition\\Analysis\\match_POI\\data\\airport.xlsx')
    df_railway=pd.read_excel('F:\\PycharmProjects\\VehicleRecognition\\Analysis\\match_POI\\data\\railway_station.xlsx')

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


    '''list形式不行'''
    '''pd_airport_location_2_column=[]
    pd_airport_location_2_column.append([pd_airport_location.loc[i,'location'].split(',') for i in range(100)])'''

    # didi_HEV_volume=data_didi_HEV.shape[0]
    airport_location_volume=pd_airport_location.shape[0]
    railway_location_volume=pd_railway_location.shape[0]


    '''airport'''
    airport_location_2_column=[]
    airport_location_2_column=[pd_airport_location.loc[i,'location'].split(',') for i in range(airport_location_volume)]

    '''railway station'''
    railway_location_2_column = []
    railway_location_2_column = [pd_railway_location.loc[i, 'location'].split(',') for i in range(railway_location_volume)]

    '''20200909
       需要将list中的str元素转为float,可以了'''
    # data= [[]]
    airport_location_done=list()

    '''airport'''
    for i in range(airport_location_volume):
        # data = list(map(eval, [airport_location_2_column[i] for i in range(100)]))
        # data[i] = list(map(eval, airport_location_2_column[i]))
        '''还是用append好些，用list的索引添加很多问题'''
        airport_location_done.append(list(map(eval, airport_location_2_column[i])))

    '''二维数组再转为DataFrame格式，可以了'''
    airport_location_done=pd.DataFrame(airport_location_done,columns=['longitude','latitude'])

    '''railway station'''
    railway_location_done = list()
    for i in range(railway_location_volume):
        railway_location_done.append(list(map(eval, railway_location_2_column[i])))
    railway_location_done = pd.DataFrame(railway_location_done, columns=['longitude', 'latitude'])

    # airport_location.split(",") #字符串转为列表有split(",")
    s1 = ','.join(str(n) for n in airport_location)
    a=1
    return airport_location_done,railway_location_done


def matchPOI():
    airport_location, railway_location=POIPorcess()
    # data_didi_HEV=dataPrePorcess()
    flag=0
    data_didi_HEV = dataPrePorcess(flag)


    # data_didi_HEV=pd.read_csv('F:\\sql data\\classifer_car_data\\example\\SHEVDC_0A101F56_vehicle_position.csv')
    data_didi_HEV_location=pd.concat([data_didi_HEV['lat_new'],data_didi_HEV['lng_new']],axis=1)

    '''airport'''
    match_points_airport=0
    for i in range(airport_location.shape[0]):
        for j in range(data_didi_HEV_location.shape[0]):
            if (data_didi_HEV_location.loc[j,'lat_new'].round(decimals=3)==airport_location.loc[i,'latitude'].round(decimals=3))\
                    and (data_didi_HEV_location.loc[j,'lng_new'].round(decimals=3)==airport_location.loc[i,'longitude'].round(decimals=3)):
                match_points_airport+=1

    '''railway station'''
    match_points_railway = 0
    for i in range(railway_location.shape[0]):
        for j in range(data_didi_HEV_location.shape[0]):
            if (data_didi_HEV_location.loc[j,'lat_new'].round(decimals=3)==railway_location.loc[i,'latitude'].round(decimals=3))\
                    and (data_didi_HEV_location.loc[j,'lng_new'].round(decimals=3)==railway_location.loc[i,'longitude'].round(decimals=3)):
                match_points_railway+=1

    return match_points_airport,match_points_railway


if __name__ == '__main__':
    starttime = datetime.datetime.now()
    divideDay()
    match_points_airport,match_points_railway=matchPOI()
    print('match_points_airport:{}\nmatch_points_railway_station:{}'.format(match_points_airport,match_points_railway))
    endtime = datetime.datetime.now()
    print('CPU time:',(endtime - starttime).seconds)





