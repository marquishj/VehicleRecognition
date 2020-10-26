# -*- coding: utf-8 -*-
# @Time    : 2020/10/26 14:37
# @Author  : Hou Jue
# @Email   : 382062104@qq.com
# @File    : POI_result_analysis_20201025_1.py
# @Software: PyCharm

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import sys
import datetime
import math

class Analysis():

    def __init__(self,id):
        # self.readData()
        self.days = ['2019-09-01', '2019-09-02', '2019-09-03', '2019-09-04', '2019-09-05', '2019-09-06', '2019-09-07',
                    '2019-09-08', '2019-09-09', '2019-09-10', '2019-09-11', '2019-09-12', '2019-09-13', '2019-09-14', '2019-09-15',
                    '2019-09-16', '2019-09-17', '2019-09-18', '2019-09-19', '2019-09-20', '2019-09-21', '2019-09-22',
                    '2019-09-23', '2019-09-24', '2019-09-25', '2019-09-26', '2019-09-27', '2019-09-28', '2019-09-29', '2019-09-30']
        self.id=id
        # self.vehicleIDs=[]
        # self.vehicle=[]

    # def __init__(self):
    #     pass


    def readVehicleID(self):
        vehicle=[]
        vehicleIDs=[]
        df = pd.read_table("F:\\sql data\\classifer_car_data\\分析结果\\2020.10.19\\POI_0.5km\\result_didi.txt")
        # searchfor = ['Vehicle ID:']
        '''行数'''
        for i in range(df.shape[0]):
            # vehicleID=df.iloc[:, 0].loc[i][:10]==['Vehicle ID:'].index.tolist()
            # vehicleID.append(df[df.iloc[:, 0].loc[i][:11]=='Vehicle ID:'].index.tolist())
            if df.iloc[:, 0].loc[i][:11]=='Vehicle ID:':
                vehicle.append(i)
                a=1
        # res = df[df['Vehicle ID:'].str.contains("<img")]
        # df[df.str.contains('|'.join(searchfor))]
        # vehicleID=df.iloc['Vehicle ID:', 0]
        for id in vehicle:
            # vehicleIDs=df.iloc[id, 0]
            vehicleIDs.append(df.iloc[id, 0][-20:])
        '''有问题：vehicle, vehicleIDs 一样？'''
        return vehicle, vehicleIDs


    def readData(self):
        df = pd.read_table("F:\\sql data\\classifer_car_data\\分析结果\\2020.10.19\\POI_0.5km\\result_didi.txt")
        return df

    '''找出所有车的范围，其实只需要找一台即可'''
    def findVehicleRange(self):
        vehicle, vehicleIDs = Analysis(0).readVehicleID()
        eachVehicle_nums = []
        eachVehicle_nums_1 = []
        eachVehicle_nums_2 = pd.DataFrame()
        # n = 0
        '''思路：找到每台车的数据的起始和终点的位置（行号）'''
        '''eachVehicle_nums_2'''
        '''然后每台车从起始和终点的位置（行号）开始找
           eachVehicle_nums_2.loc[i,0]
           eachVehicle_nums_2.loc[i,1]'''
        for i in range(len(vehicle)):
            # eachVehicle_nums.append(vehicle[i + 1] - vehicle[i])
            # eachVehicle_nums_1.append([vehicle[i], vehicle[i + 1]])
            '''注意最后一个元素会溢出'''
            if i==len(vehicle)-1:
                eachVehicle_nums_2.loc[i-1, 0] = vehicle[i-1]
                eachVehicle_nums_2.loc[i-1, 1] = vehicle[i]
                break
            else:
                eachVehicle_nums_2.loc[i, 0] = vehicle[i]
                eachVehicle_nums_2.loc[i, 1] = vehicle[i + 1]
            # n += 1
            pass
        return eachVehicle_nums_2



    def statistcisData(self):
        df=self.readData()
        eachVehicle_nums_2=self.findVehicleRange()
        '''第1台车'''
        # start=eachVehicle_nums_2[0][0]
        start=eachVehicle_nums_2.loc[self.id][0]
        # end=eachVehicle_nums_2[1][0]
        end=eachVehicle_nums_2.loc[self.id][1]
        # vehicle, vehicleIDs = Analysis(0).readVehicleID()

        # df_start =self.vehicleID
        # df_start='Vehicle ID: SHEVDC_012V252I.csv'
        # df_end='Vehicle ID: SHEVDC_02205P41.csv'
        '''需要处理的数据在 df.iloc[:, 0]'''
        # data=df.iloc[:, 0]['2019-09-01']

        points_POI_match_airport = []
        points_POI_match_railway = []

        '''还需要针对VehicleID找到所有天数'''
        vehicleIDs = Analysis(0).readVehicleID()
        '''偷懒了，省去一次调用计算步骤
            self似乎不行'''

        for day in self.days:
            # position=df[df.iloc[:, 0] == day].index.tolist()
            # position=df[df.iloc[int(start):int(end), 0] == day].index.tolist()
            position=df.iloc[int(start):int(end), 0][df.iloc[int(start):int(end), 0]==day].index.tolist()
            # length=len(position)i
            # index=position[0]

        # for i in position[0]:
        #     if df.iloc[position[index]+1, 0]=='Cannot find this day from datasets.':
            if df.iloc[position[0]+1, 0]=='Cannot find this day from datasets.':
                points_POI_match_airport.append('null')
                points_POI_match_railway.append('null')

                continue
            else:
                '''怎么保存数据 1个表头（时间）2个值（airport，railway）'''
                # position_POI_match_airport=df.iloc[position[index]+1, 0]
                # position_POI_match_railway=df.iloc[position[index]+2, 0]
                position_POI_match_airport=df.iloc[position[0]+1, 0]
                position_POI_match_railway=df.iloc[position[0]+2, 0]

                points_POI_match_airport.append(int(position_POI_match_airport[-1]))
                points_POI_match_railway.append(int(position_POI_match_railway[-1]))

                # points_POI_match_airport=int(position_POI_match_airport[-1])
                # points_POI_match_railway=int(position_POI_match_railway[-1])
                # table=pd.DataFrame()
                # table=1
                # table = pd.DataFrame(table)
                # table.columns=list(day)
                # table=pd.DataFrame(points_POI_match_airport, columns=day)
                # a=1

            # find(df=='match_points_airport:')
            # res = df[df['match_points_airport:'].str.contains("<img")]
        return points_POI_match_airport,points_POI_match_railway


if __name__ == '__main__':
    '''构造函数只能有一个？'''
    vehicle, vehicleIDs = Analysis(0).readVehicleID()
    # for id in vehicle:
    '''不用找行号，按序号寻找即可'''
    for id in range(len(vehicle)):
        # vehicle_row= vehicle[id]
        # vehicleIDs=['Vehicle ID: SHEVDC_012V252I.csv']
        # for vehicleID in vehicleIDs:
        '''参数为list'''
        # vehicleID=[vehicleID]
        # analysis_1=Analysis(vehicleID)
        # analysis_1=Analysis(vehicle)
        analysis_1=Analysis(id)
        '''每一台车分析，出来30天的数据'''
        points_POI_match_airport,points_POI_match_railway=analysis_1.statistcisData()
        print('vehicleID: {}'.format(id))
        # print('vehicleID: {}'.format(vehicleID))
        print('points_POI_match_airport: {}'.format(points_POI_match_airport))
        print('points_POI_match_railway: {}'.format(points_POI_match_railway))

    # pass
