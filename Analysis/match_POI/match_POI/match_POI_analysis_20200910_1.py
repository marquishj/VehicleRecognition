import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
'''@Author Hou Jue
   20200910'''

def dataPrePorcess():

    pcev_list = os.listdir("D:\\data\\私家车\\纯电\\")
    pcphev_list= os.listdir("D:\\data\\私家车\\混动\\")
    didi_list = os.listdir("D:\\data\\网约车\\")
    taxi_list = os.listdir("D:\\data\\出租车\\")

    '''didi-EV'''
    for didi_id in didi_list:
        data_didi_HEV = pd.read_csv('D:\\data\\网约车\\' + didi_list)
        print("missing data" + didi_list, data_didi_HEV.isnull().sum())
        print("rows in dataset" + didi_list, data_didi_HEV.shape[0])
        data_didi_HEV = data_didi_HEV.dropna()
        print("missing data-processed" + didi_list, data_didi_HEV.isnull().sum())
        print("rows in dataset-processed" + didi_list, data_didi_HEV.shape[0])

    return  data_didi_HEV

# df = pd.read_csv('...\\....csv')

def POIPorcess():

    # data_didi_HEV=dataPrePorcess()

    data_didi_HEV=pd.read_csv('F:\\sql data\\classifer_car_data\\example\\data\\SHEVDC_0A101F56_vehicle_data.csv')
    data_didi_HEV['datatime'] = pd.to_datetime(data_didi_HEV['datatime']).apply(lambda x: x.date())
    data_didi_HEV['datatime'] = pd.to_datetime(data_didi_HEV['datatime']).dt.date



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

    didi_HEV_volume=data_didi_HEV.shape[0]
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
    data=list()
    for i in range(100):
        # data = list(map(eval, [airport_location_2_column[i] for i in range(100)]))
        # data[i] = list(map(eval, airport_location_2_column[i]))
        '''还是用append好些，用list的索引添加很多问题'''
        data.append(list(map(eval, airport_location_2_column[i])))

    '''二维数组再转为DataFrame格式，可以了'''

    pd_data=pd.DataFrame(data,columns=['longitude','latitude'])

    # airport_location.split(",") #字符串转为列表有split(",")
    s1 = ','.join(str(n) for n in airport_location)
    a=1


if __name__ == '__main__':
    POIPorcess()




