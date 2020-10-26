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

    def __init__(self,vehicleID):
        # self.readData()
        self.days = ['2019-09-01', '2019-09-02', '2019-09-03', '2019-09-04', '2019-09-05', '2019-09-06', '2019-09-07',
                    '2019-09-08', '2019-09-09', '2019-09-10', '2019-09-11', '2019-09-12', '2019-09-13', '2019-09-14', '2019-09-15',
                    '2019-09-16', '2019-09-17', '2019-09-18', '2019-09-19', '2019-09-20', '2019-09-21', '2019-09-22',
                    '2019-09-23', '2019-09-24', '2019-09-25', '2019-09-26', '2019-09-27', '2019-09-28', '2019-09-29', '2019-09-30']
        self.vehicleID=vehicleID


    def readData(self):
        df = pd.read_table("F:\\sql data\\classifer_car_data\\分析结果\\2020.10.19\\POI_0.5km\\result_didi.txt")
        return df

    def statistcisData(self):
        df=self.readData()
        df_start =self.vehicleID
        # df_start='Vehicle ID: SHEVDC_012V252I.csv'
        # df_end='Vehicle ID: SHEVDC_02205P41.csv'
        '''需要处理的数据在 df.iloc[:, 0]'''
        # data=df.iloc[:, 0]['2019-09-01']
        position=df[df.iloc[:, 0] == '2019-09-01'].index.tolist()
        position_POI_match_airport=df.iloc[position[0]+1, 0]
        position_POI_match_railway=df.iloc[position[0]+2, 0]
        points_POI_match_airport=int(position_POI_match_airport[-1])
        points_POI_match_railway=int(position_POI_match_railway[-1])

        for day in self.days:
            # find(df=='match_points_airport:')
            res = df[df['match_points_airport:'].str.contains("<img")]
        return points_POI_match_airport,points_POI_match_railway


if __name__ == '__main__':

    vehicleIDs='Vehicle ID: SHEVDC_012V252I.csv'
    for vehicleID in vehicleIDs:
        analysis_1=Analysis(vehicleID)
        '''每一台车分析，出来30天的数据'''
        points_POI_match_airport,points_POI_match_railway=analysis_1.statistcisData()

    pass
