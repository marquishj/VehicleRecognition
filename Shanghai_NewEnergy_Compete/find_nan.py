import pandas as pd
import numpy  as py
import matplotlib.pyplot as plt
import matplotlib
import os
BEV_list = os.listdir("/home/kesci/input/data2737/BEV")
PHEV_list = os.listdir("/home/kesci/input/data2737/PHEV")

Num = len(BEV_list)
i = 0
for i in range(Num):
    if i <= Num:
        BEV_id = BEV_list[i]
        v1=pd.read_csv("/home/kesci/input/data2737/BEV/"+BEV_id)
        print(BEV_id,v1['summileage'].isna().sum())

Num = len(BEV_list)
i = 0
for i in range(Num):
    if i <= Num:
        BEV_id = BEV_list[i]
        v1=pd.read_csv("/home/kesci/input/data2737/BEV/"+BEV_id)
        print(BEV_id,v1['summileage'].isna().sum())