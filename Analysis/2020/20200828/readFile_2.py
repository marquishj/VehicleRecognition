import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
import os

'''Hou Jue 2020.8.28'''

def dataPrePorcess():

    pcev_list = os.listdir("D:\\data\\私家车\\纯电\\")
    pcphev_list= os.listdir("D:\\data\\私家车\\纯电\\")
    didi_list = os.listdir("D:\\data\\网约车\\")
    taxi_list = os.listdir("D:\\data\\出租车\\")


    '''private car-EV'''
    for pcev_id in pcev_list:
        data_private_EV=pd.read_csv('D:\\data\\私家车\\纯电\\'+pcev_id)
        print("missing data"+pcev_id,data_private_EV.isnull().sum())
        print("rows in dataset"+pcev_id,data_private_EV.shape[0])
        data_private_EV=data_private_EV.dropna()
        print("missing data-processed"+pcev_id,data_private_EV.isnull().sum())
        print("rows in dataset-processed"+pcev_id,data_private_EV.shape[0])

    '''private car-PHEV'''
    for pcphev_id in pcphev_list:
        data_private_PHEV=pd.read_csv('D:\\data\\私家车\\混动\\'+pcphev_id)
        print("missing data"+pcphev_id,data_private_PHEV.isnull().sum())
        print("rows in dataset"+pcphev_id,data_private_PHEV.shape[0])
        data_private_PHEV=data_private_PHEV.dropna()
        print("missing data-processed"+pcphev_id,data_private_PHEV.isnull().sum())
        print("rows in dataset-processed"+pcphev_id,data_private_PHEV.shape[0])

    '''didi car hailing-PHEV'''
    for didi_id in didi_list:
        data_didi_PHEV=pd.read_csv('D:\\data\\网约车'+didi_id)
        print("missing data"+didi_list,data_didi_PHEV.isnull().sum())
        print("rows in dataset"+didi_list,data_didi_PHEV.shape[0])
        data_didi_PHEV=didi_list.dropna()
        print("missing data-processed"+didi_list,data_didi_PHEV.isnull().sum())
        print("rows in dataset-processed"+didi_list,data_didi_PHEV.shape[0])

    '''taxi-EV'''
    for taxi_id in taxi_list:
        data_taxi_EV=pd.read_csv('D:\\data\\出租车\\'+taxi_id)
        print("missing data"+taxi_id,data_taxi_EV.isnull().sum())
        print("rows in dataset"+taxi_id,data_taxi_EV.shape[0])
        data_taxi_EV=data_taxi_EV.dropna()
        print("missing data-processed"+taxi_id,data_taxi_EV.isnull().sum())
        print("rows in dataset-processed"+taxi_id,data_taxi_EV.shape[0])



    return 0


if __name__ == '__main__':
    dataPrePorcess()

