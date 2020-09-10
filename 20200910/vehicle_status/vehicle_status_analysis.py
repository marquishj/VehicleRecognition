# -*- coding: utf-8 -*-


import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.animation import ArtistAnimation

'''Hou Jue
    找到01 车辆启动和 02 熄火状态占一天/周/月的百分比
'''


def find_nonzero(x,f):
    x = list(x)
    if f == 0:
        for i in range(len(x)):
            if x[i]>0:
                break
        t=x[i]
    else:
        for i in range(len(x)):
            if x[-(x+1)]>0:
                break
        t=x[-(x+1)]
    return t

def find_miles(x):
    ori = find_nonzero(x,0)
    des = find_nonzero(x,1)
    distance = des - ori
    return distance


'''didi car-hailing PHEV'''
didi_list = os.list("data/didi")
sum_didi = []  ###################
run1 = []
run2 = []
run3 = []
a = lambda x: sum(x == 1)
b = lambda x: sum(x == 2)
c = lambda x: sum(x == 3)
for didi_id in didi_list:
    one_vehicle = pd.read_csv("/data/didi/" + didi_id)
    one_vehicle['time'] = pd.to_datetime(one_vehicle['time']).apply(lambda x: x.date())  # year,month,day
    agg = one_vehicle.groupby('time').agg({"vehiclestatus": [a, b, c]})
    agg = agg.apply(lambda x: x / sum(x), axis=1)
    name = agg.columns
    r1 = list(agg[name[0]])
    r2 = list(agg[name[1]])
    r3 = list(agg[name[2]])

    run1 += r1
    run2 += r2
    run3 += r3
plt.hist(np.array(sum_didi).T[1] / 360, bin=30)