# -*- coding: utf-8 -*-


import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.animation import ArtistAnimation

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
soc_didi=[]
didi_list=os.list("data/didi")
for didi_id in didi_list:
    one_vehicle = pd.read_csv("/data/didi/"+didi_id)
    mean = np.mean(one_vehicle['soc'])
    std = np.std(one_vehicle['soc'])
    soc_didi.append((mean,std))

'''PCHEV'''
soc_pchev=[]
pchev_list=os.list("data/pchev")
for pchev_id in pchev_list:
    one_vehicle = pd.read_csv("/data/pchev/"+pchev_id)
    mean = np.mean(one_vehicle['soc'])
    std = np.std(one_vehicle['soc'])
    soc_didi.append((mean,std))

'''taxi'''
soc_taxi=[]
taxi_list=os.list("data/taxi")
for taxi_id in taxi_list:
    one_vehicle = pd.read_csv("/data/taxi/"+taxi_id)
    mean = np.mean(one_vehicle['soc'])
    std = np.std(one_vehicle['soc'])
    soc_didi.append((mean,std))

'''PCEV'''
soc_pcev=[]
pcev_list=os.list("data/pcev")
for pcev_id in pcev_list:
    one_vehicle = pd.read_csv("/data/pcev/"+pcev_id)
    mean = np.mean(one_vehicle['soc'])
    std = np.std(one_vehicle['soc'])
    soc_didi.append((mean,std))

d1 = np.array(soc_taxi).T
d2 = np.array(soc_pcev).T
d3 = np.array(soc_didi).T
d4 = np.array(soc_pchev).T

plt.figure(figsize=(10,10))   
plt.subplot(221)
plt.xlim(0,90)
plt.ylim(0,35)
plt.plot(d1[0],d1[1],'bo',d2[0],d2[1],'go')
plt.ylabel("standard devation of monthly soc")
plt.xlabel("mean of monthly soc")
plt.legend(("taxi","EV Private Car").loc="upper left")

plt.subplot(222)
plt.xlim(0,90)
plt.ylim(0,35)
plt.plot(d1[0],d1[1],'bo',d4[0],d4[1],'go')
plt.ylabel("standard devation of monthly soc")
plt.xlabel("mean of monthly soc")
plt.legend(("taxi","PHEV Private Car").loc="upper left")

plt.subplot(223)
plt.xlim(0,90)
plt.ylim(0,35)
plt.plot(d3[0],d3[1],'bo',d2[0],d2[1],'go')
plt.ylabel("standard devation of monthly soc")
plt.xlabel("mean of monthly soc")
plt.legend(("didi car-hailing PHEV","EV Private Car",).loc="upper left")

plt.subplot(224)
plt.xlim(0,90)
plt.ylim(0,35)
plt.plot(d3[0],d3[1],'bo',d4[0],d4[1],'go')
plt.ylabel("standard devation of monthly soc")
plt.xlabel("mean of monthly soc")
plt.legend(("didi car-hailing PHEV","PHEV Private Car",).loc="upper left")
