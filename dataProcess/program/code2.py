# -*- coding: utf-8 -*-
############# daily driving distance

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
            if x[-(i+1)]>0:
                break
        t=x[-(i+1)]
    return t 

def find_miles(x):
    ori = find_nonzero(x,0)
    des = find_nonzero(x,1)
    distance = des - ori
    return distance

'''didi car-hailing PHEV'''
didi_list = os.listdir("data/didi")
ddv_didi=[]
for didi_id in didi_list:
    one_vehicle = pd.read_csv("/data/didi/"+didi_id)
    one_vehicle['time'] = pd.to_datetime(one_vehicle['time']).apply(lambda x:x.date()) #year,month,day
    agg = one_vehicle.groupby('time').agg({"summileage":find_miles})  # list of daily distance of 30 days
    f = (np.mean(list(agg["summileage"])),np.std(list(agg["summileage"])))
    ddv_didi.append(f)

'''taxi EV '''
taxi_list = os.listdir("data/taxi")
ddv_taxi=[]
for taxi_id in taxi_list:
    one_vehicle = pd.read_csv("/data/taxi/"+taxi_id)
    one_vehicle['time'] = pd.to_datetime(one_vehicle['time']).apply(lambda x:x.date()) #year,month,day
    agg = one_vehicle.groupby('time').agg({"summileage":find_miles})
    f = (np.mean(list(agg["summileage"])),np.std(list(agg["summileage"])))
    ddv_taxi.append(f)

'''pchev'''
pchev_list = os.listdir("data/pchev")
ddv_pchev=[]
for pchev_id in pchev_list:
    one_vehicle = pd.read_csv("/data/pchev/"+taxi_id)
    one_vehicle['time'] = pd.to_datetime(one_vehicle['time']).apply(lambda x:x.date()) #year,month,day
    agg = one_vehicle.groupby('time').agg({"summileage":find_miles})
    f = (np.mean(list(agg["summileage"])),np.std(list(agg["summileage"])))
    ddv_pchev.append(f)

'''phev'''
pcev_list = os.listdir("data/pcev")
ddv_pcev=[]
for pcev_id in pcev_list:
    one_vehicle = pd.read_csv("/data/pcev/"+taxi_id)
    one_vehicle['time'] = pd.to_datetime(one_vehicle['time']).apply(lambda x:x.date()) #year,month,day
    agg = one_vehicle.groupby('time').agg({"summileage":find_miles})
    f = (np.mean(list(agg["summileage"])),np.std(list(agg["summileage"])))
    ddv_pcev.append(f)

ddv_taxi = [i for i in ddv_taxi if i[0]>0]

np.save("ddv_didi.py",np.array(ddv_didi))
np.save("ddv_taxi.py",np.array(ddv_taxi))
np.save("ddv_pchev.py",np.array(ddv_pchev))
np.save("ddv_pcev.py",np.array(ddv_pcev))

d1=np.load("ddv_taxi.py.npy").T
d2=np.load("ddv_pcev.py.npy").T
d3=np.load("ddv_didi.py.npy").T
d4=np.load("ddv_pchev.py.npy").T


plt.figure(figsize=(15,10))   
plt.subplot(231)
plt.xlim(0,500)
plt.ylim(0,225)
plt.plot(d1[0],d1[1],"#D9534F",d2[0],d2[1],"#F0AD4E")
plt.ylabel("standard devation of daily distance")
plt.xlabel("mean of daily driving distance")
plt.legend(("taxi","EV Private Car"),loc="upper left")

plt.subplot(232)
plt.xlim(0,500)
plt.ylim(0,225)
plt.plot(d1[0],d1[1],'#D9534F',d4[0],d4[1],'#F0AD4E')
plt.ylabel("standard devation of daily distance")
plt.xlabel("mean of daily driving distance")
plt.legend(("taxi","PHEV Private Car"),loc="upper left")

plt.subplot(233)
plt.xlim(0,500)
plt.ylim(0,225)
plt.plot(d1[0],d1[1],'#D9534F',d3[0],d3[1],'#F0AD4E')
plt.ylabel("standard devation of daily distance")
plt.xlabel("mean of daily driving distance")
plt.legend(("taxi","didi car-hailing PHEV"),loc="upper left")

plt.subplot(234)
plt.xlim(0,500)
plt.ylim(0,225)
plt.plot(d3[0],d3[1],'#D9534F',d2[0],d2[1],'#F0AD4E')
plt.ylabel("standard devation of daily distance")
plt.xlabel("mean of daily driving distance")
plt.legend(("didi car-hailing PHEV","EV Private Car",),loc="upper left")

plt.subplot(235)
plt.xlim(0,500)
plt.ylim(0,225)
plt.plot(d3[0],d3[1],'#D9534F',d4[0],d4[1],'#F0AD4E')
plt.ylabel("standard devation of daily distance")
plt.xlabel("mean of daily driving distance")
plt.legend(("didi car-hailing PHEV","PHEV Private Car",),loc="upper left")

plt.subplot(236)
plt.xlim(0,500)
plt.ylim(0,225)
plt.plot(d2[0],d2[1],'#D9534F',d4[0],d4[1],'#F0AD4E')
plt.ylabel("standard devation of daily distance")
plt.xlabel("mean of daily driving distance")
plt.legend(("EV Private Car","PHEV Private Car",),loc="upper left")