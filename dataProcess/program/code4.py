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
sum_didi=[] ###################
run1 = []
run2 = []
run3 = []
a = lambda x:sum(x==1)
b = lambda x:sum(x==2)
c = lambda x:sum(x==3)
for didi_id in didi_list:
    one_vehicle = pd.read_csv("/data/didi/"+didi_id)
    one_vehicle['time'] = pd.to_datetime(one_vehicle['time']).apply(lambda x:x.date()) #year,month,day
    agg = one_vehicle.groupby('time').agg({"runmodel":[a,b,c]})
    agg = agg.apply(lambda x: x/sum(x),axis=1)
    name = agg.columns
    r1 = list(agg[name[0]])
    r2 = list(agg[name[1]])
    r3 = list(agg[name[2]])
    
    run1 += r1
    run2 += r2
    run3 += r3
plt.hist(np.array(sum_didi).T[1]/360,bin=30)


'''taxi EV '''
taxi_list = os.listdir("data/taxi")
sum_taxi=[] ###################
run1_taxi = []
run2_taxi = []
run3_taxi = []
for taxi_id in taxi_list:
    one_vehicle = pd.read_csv("/data/taxi/"+taxi_id)
    time = one_vehicle.columns[0]  
    one_vehicle['time'] = pd.to_datetime(one_vehicle['time']).apply(lambda x:x.date()) #year,month,day
    agg = one_vehicle.groupby('time').agg({"runmodel":[a,b,c]})
    agg = agg.apply(lambda x: x/sum(x),axis=1)
    name = agg.columns
    r1 = list(agg[name[0]])
    r2 = list(agg[name[1]])
    r3 = list(agg[name[2]])    
    run1_taxi += r1
    run2_taxi += r2
    run3_taxi += r3
sum_taxi2 = [i for i in sum_taxi if i>0]

'''private car PHEV '''
pchev_list = os.listdir("data/pchev")
sum_pchev=[] ###################
run1_pchev = []
run2_pchev = []
run3_pchev = []
for pchev_id in pchev_list:
    one_vehicle = pd.read_csv("/data/pchev/"+pchev_id)
    time = one_vehicle.columns[0]  
    one_vehicle['time'] = pd.to_datetime(one_vehicle['time']).apply(lambda x:x.date()) #year,month,day
    agg = one_vehicle.groupby('time').agg({"runmodel":[a,b,c]})
    agg = agg.apply(lambda x: x/sum(x),axis=1)
    name = agg.columns
    r1 = list(agg[name[0]])
    r2 = list(agg[name[1]])
    r3 = list(agg[name[2]])    
    run1_pchev += r1
    run2_pchev += r2
    run3_pchev += r3
np.means([i for i in run3 if np.isnan(i)==0])
plt.hist(run3,bins=100,density=True)
plt.hist(run3_pchev,bins=100,density=True)

'''private car PHEV '''
plt.subplot(2,1,1)
plt.ylim(0,11)
plt.hist(run1_pchev,bins=100,density=True)
plt.hist(run2_pchev,bins=100,density=True)
plt.legend(("runmodel1","runmodel2"))
plt.xlabel("percentage of using one runmodel")
plt.ylabel("density")

'''didi car-hailing PHEV'''
plt.subplot(2,1,2)
plt.ylim(0,11)
plt.hist(run1,bins=100,density=True)
plt.xlabel("percentage of using one runmodel")
plt.ylabel("density")
plt.hist(run2,bins=100,density=True)

'''private car EV '''
pcev_list = os.listdir("data/pchev")
sum_pcev=[] ###################
run1_pcev = []
run2_pcev = []
run3_pcev = []
for pcev_id in pcev_list:
    one_vehicle = pd.read_csv("/data/pchev/"+pcev_id)
    time = one_vehicle.columns[0]  
    one_vehicle['time'] = pd.to_datetime(one_vehicle['time']).apply(lambda x:x.date()) #year,month,day
    agg = one_vehicle.groupby('time').agg({"runmodel":[a,b,c]})
    agg = agg.apply(lambda x: x/sum(x),axis=1)
    name = agg.columns
    r1 = list(agg[name[0]])
    r2 = list(agg[name[1]])
    r3 = list(agg[name[2]])    
    run1_pcev += r1
    run2_pcev += r2
    run3_pcev += r3

plt.figure(figsize=(10,10))   
'''taxi'''
plt.subplot(221)
plt.xlim(0,30)
plt.ylim(0,40)
plt.hist(np.array(sum_taxi).T[0],bins=30)
plt.ylabel("Taxi Number")
plt.xlabel("monthly charge frequency")
plt.legend(("taxi").loc="upper left")

'''pcev'''
plt.subplot(222)
plt.xlim(0,30)
plt.ylim(0,40)
plt.hist(np.array(sum_pcev).T[0],bins=30)
plt.xlabel("daily driving distance")
plt.ylabel("frequency")
plt.legend(("EV private car").loc="upper left")

'''didi'''
plt.subplot(223)
plt.xlim(0,30)
plt.ylim(0,40)
plt.hist(np.array(sum_didi).T[0],bins=30)
plt.xlabel("daily driving distance")
plt.ylabel("frequency")
plt.legend(("didi").loc="upper left")

'''pchev'''
plt.subplot(224)
plt.xlim(0,30)
plt.ylim(0,40)
plt.hist(np.array(sum_pchev).T[0],bins=30)
plt.xlabel("daily driving distance")
plt.ylabel("frequency")
plt.legend(("PCHEV").loc="upper left")