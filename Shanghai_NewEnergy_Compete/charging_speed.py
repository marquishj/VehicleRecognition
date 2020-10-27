# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 22:40:52 2020

@author: zhang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(1,421540)
#x = np.arange(1, 177038)
id = 'SHEVDC_144S705J.csv'
# id = 'SHEVDC_0E5G6T33.csv'
# id = 'SHEVDC_1C2L2J5R.csv'
total = pd.read_csv('/home/kesci/input/data2737/BEV/' + id)
data_didi_HEV = pd.read_excel('F:\Marquez\任务：Carsharing\新能源车辆数据中心\\BEV样本数据.xlsx')
licheng = total['summileage']
LC = list(licheng)



total = total[total['chargestatus'] != 254] 
A = total[total['chargestatus'] == 1] 
# print(total.shape[0])      # 共421539条数据，充电状态为1的有178631条
charging_index = A.index        # 充电状态为1的index
STOP = []
START =[charging_index[0]]
for i in range(1, len(charging_index) ):
    if charging_index[i] - charging_index[i - 1] != 1:        # index不连续
        stop_charge = charging_index[i - 1]                   # 最后一个1为停止充电的点
        STOP.append(stop_charge)
        START.append(charging_index[i])
if (len(START) - len(STOP) == 1):                       # 补齐结尾
    STOP.append(charging_index[-1])
# print(STOP)
# print(START)
# print(len(STOP))
# print(len(START))
    
DF = []
for j in range(len(STOP)):
    df = total[START[j] : STOP[j]]
    DF.append(df)
print(len(DF))

# 计算充电时间秒数
import datetime
DIF = []
for m in range(len(DF)):
    ks = datetime.datetime.strptime(DF[m]['datatime'].iloc[0], '%Y-%m-%d %H:%M:%S')
    js = datetime.datetime.strptime(DF[m]['datatime'].iloc[-1], '%Y-%m-%d %H:%M:%S')
    dif = (js - ks).seconds
    DIF.append(dif)
print(DIF)



# 计算SOC差值
SOC_dif = []
for n in range(len(DF)):
    ks = DF[n]['soc'].iloc[0]
    js = DF[n]['soc'].iloc[-1]
    dif = js - ks
    SOC_dif.append(dif)
print(SOC_dif)


# 计算充电速度
SPEED = []
for p in range(len(DF)):
    speed = (SOC_dif[p] /  DIF[p]) * 3600
    SPEED.append(speed)
print(SPEED)
