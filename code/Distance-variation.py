import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns

from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
import os

# didi_list=os.listdir("data/didi")
didi_list=os.listdir(" ")
pcev_list=os.listdir(" ")
pcphev_list=os.listdir(" ")
taxi_list=os.listdir(" ")

def find_nonzero(x,f):
    x=list(x)
    if f==0:
        for i in range(len(x)):
            if x[i]>0:
                break
    else:
        for i in range(len(x)):
            if x[-(i+1)]>0:
                break
        t=x[-(i+1)]
    return t


def find_miles(x):
    o=find_nonzero(x,0)
    d=find_nonzero(x,1)
    dis=d-o
    return dis


'''didi car-hailing PHEV'''
sum_didi=[]
for didi_id in didi_list:
    v1=pd.read_csv(" "+didi_id)
    v1['time']=pd.to_datetime(v1['time']).apply(lambda x:x.date())
    agg=v1.groupby('time').agg({'summileage':find_miles()})
    [sum_didi.append(i) for i in list(agg['summileage'])]


'''private car EV '''
sum_pcev=[]
for pcev_id in pcev_list:
    v1=pd.read_csv(" "+pcev_id)
    v1['time']=pd.to_datetime(v1['time']).apply(lambda x:x.date())
    agg=v1.groupby('time').agg({'summileage':find_miles()})
    [sum_pcev.append(i) for i in list(agg['summileage'])]

'''private car PHEV '''
sum_pcphev=[]
for pcphev_id in pcphev_list:
    v1 = pd.read_csv(" " + pcphev_id)
    v1['time']=pd.to_datetime(v1['time']).apply(lambda x:x.date())
    agg=v1.groupby('time').agg({'summileage':find_miles()})
    [sum_pcphev.append(i) for i in list(agg['summileage'])]

'''taxi EV '''
sum_taxi=[]
for taxi_id in taxi_list:
    v1=pd.read_csv("/"+taxi_id)
    v1['time']=pd.to_datetime(v1['time']).apply(lambda x:x.date())
    agg=v1.groupby('time').agg({'summileage':find_miles()})
    [sum_pcphev.append(i) for i in list(agg['summileage'])]


plt.figure(figsize=(10,10))

'''didi car-hailing PHEV'''
plt.subplot(221)
plt.xlim(0,600)
plt.ylim(0,600)
plt.hist(sum_didi,bins=40)
plt.ylabel('frequency')
plt.xlabel('daily driving distance')
plt.text(400,500,"Mean="+str(round(np.mean(sum_didi))))
plt.text(400,500,"Std="+str(round(np.std(sum_didi))))
plt.legend(('didi','Variance='+str(np.var(sum_didi))),loc="upper left")

'''private car EV '''
plt.subplot(222)
plt.xlim(0,600)
plt.ylim(0,600)
plt.hist(sum_didi,bins=40)
plt.ylabel('frequency')
plt.xlabel('daily driving distance')
plt.text(400,500,"Mean="+str(round(np.mean(sum_pcev))))
plt.text(400,500,"Std="+str(round(np.std(sum_pcev))))
plt.legend(('EV private car','Variance='+str(np.var(sum_pcev))),loc="upper left")

'''private car PHEV '''
plt.subplot(223)
plt.xlim(0,600)
plt.ylim(0,600)
plt.hist(sum_didi,bins=40)
plt.ylabel('frequency')
plt.xlabel('daily driving distance')
plt.text(400,500,"Mean="+str(round(np.mean(sum_pcphev))))
plt.text(400,500,"Std="+str(round(np.std(sum_pcphev))))
plt.legend(('PHEV private car','Variance='+str(np.var(sum_pcphev))),loc="upper left")

'''taxi EV '''
plt.subplot(224)
plt.xlim(0,600)
plt.ylim(0,600)
plt.hist(sum_didi,bins=40)
plt.ylabel('frequency')
plt.xlabel('daily driving distance')
plt.text(400,500,"Mean="+str(round(np.mean(sum_taxi))))
plt.text(400,500,"Std="+str(round(np.std(sum_taxi))))
plt.legend(('EV taxi','Variance='+str(np.var(sum_taxi))),loc="upper left")


