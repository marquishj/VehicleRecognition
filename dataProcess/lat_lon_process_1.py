# Python画点或者曲线的轨迹，并保存为动图。
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd




data=pd.read_csv('F:\sql data\classifer_car_data\example\SHEVDC_0A101F56_vehicle_position.csv')




lat=data['latitude']
lon=data['longitude']


'''{Long:106.652024,Lat:26.617221},'''

dot_0=[]
dot_1=[]
dot_2=[]
dot_3=[]
for i in range(len(lat)):
    dot_0.append("{Long:")
    dot_1.append(",")
    dot_2.append("Lat:")
    dot_3.append("},")

dot_0 = np.transpose(dot_0)

data_processed=pd.DataFrame({'dot_0':dot_0,
                           'longitude': lon,
                            'dot_1':dot_1,
                            'dot_2':dot_2,
                            'latitude':lat,
                            'dot_3':dot_3})


data_processed.to_excel('F:\\sql data\\classifer_car_data\\example\\baidu_data_processed_1.xlsx')


'''[118.626495,32.05714],'''
dot_gaode_0=[]
dot_gaode_1=[]
dot_gaode_2=[]

for i in range(len(lat[:674])):
    dot_gaode_0.append("[")
    dot_gaode_1.append(",")
    dot_gaode_2.append("],")
    # dot_gaode_3.append("}")

dot_0 = np.transpose(dot_0)

gaode_data_processed_1=pd.DataFrame({'dot_0':dot_gaode_0,
                           'longitude': lon[:674],
                            'dot_1':dot_gaode_1,
                            'latitude':lat[:674],
                            'dot_3':dot_gaode_2})

gaode_data_processed_1.to_excel('F:\\sql data\\classifer_car_data\\example\\gaode_data_processed_1.xlsx')