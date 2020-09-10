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

    pchev_list = os.listdir("D:\\data\\私家车\\纯电\\")

    for pchev_id in pchev_list:

        data_private_PHEV=pd.read_csv('D:\\data\\私家车\\纯电\\SHEVDC_0BN4O461.csv')
        print("missing data",data_private_PHEV.isnull().sum())
        print("rows in dataset",data_private_PHEV.shape[0])

        data_private_PHEV=data_private_PHEV.dropna()
        print("missing data-processed",data_private_PHEV.isnull().sum())
        print("rows in dataset-processed",data_private_PHEV.shape[0])






    '''PCHEV'''
    # pchev_list=[]
    soc_pchev=[]
    for pchev_id in pchev_list:
        one_vehicle = pd.read_csv("D:\\data\\私家车\\纯电\\"+pchev_id)
        mean = np.mean(one_vehicle['soc'])
        std = np.std(one_vehicle['soc'])
        soc_pchev.append((mean,std))





    return soc_pchev


if __name__ == '__main__':
    soc_pchev=dataPrePorcess()

