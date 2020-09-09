# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


def find_nonzero(x,f):
    x = list(x)
    if f == 0:
        for i in range(len(x)):
            if x[i]>0:     #从第一个计算
                break
        t=x[i]
    else:
        for i in range(len(x)):
            if x[-(i+1)]>0: #从最后一个计算
                break
        t=x[-(i+1)]
    return t 

def find_miles(x):
    ori = find_nonzero(x,0)
    des = find_nonzero(x,1)
    distance = des - ori
    return distance


'''didi car-hailing PHEV'''
# didi_list = os.listdir("F:\\PycharmProjects\\VehicleRecognition\\dataProcess\\program\\SHEVDC_0A101F56_vehicle_data.csv")
plt.figure(figsize=(15,10)) 
Num = 5
i = 0
# for didi_id in didi_list:
# i += 1
# if i <= Num:
    # one_vehicle = pd.read_csv("/data/didi/"+didi_id)
one_vehicle = pd.read_csv("F:\\PycharmProjects\\VehicleRecognition\\dataProcess\\program\\SHEVDC_0A101F56_vehicle_data.csv")
# one_vehicle = pd.read_csv("SHEVDC_0A101F56_vehicle_data.csv")
one_vehicle['datatime'] = pd.to_datetime(one_vehicle['datatime']).apply(lambda x:x.date()) #year,month,day
# one_vehicle['datatime'] = pd.to_datetime(one_vehicle['datatime']).map(lambda x:x.date()) #year,month,day
agg = one_vehicle.groupby('datatime').aggregate({'summileage':find_miles})  # list of daily distance of 30 days
daily_miles = list(agg["summileage"])
plt.plot(range(1,30),daily_miles)
plt.show()
