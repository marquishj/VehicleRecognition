import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
'''Hou Jue
   20200909'''



# df = pd.read_csv('...\\....csv')

df_airport=pd.read_excel('F:\\PycharmProjects\\VehicleRecognition\\Analysis\\match_POI\\data\\airport.xlsx')
df_railway_station=pd.read_excel('F:\\PycharmProjects\\VehicleRecognition\\Analysis\\match_POI\\data\\railway_station.xlsx')

df_airport_location=df_airport.loc[:,'location']
airport_location=list(df_airport_location)

pd_airport_location=pd.DataFrame(df_airport_location)

pd_airport_location.loc[0,'location'].split(',')

'''list形式不行'''
'''pd_airport_location_2_column=[]
pd_airport_location_2_column.append([pd_airport_location.loc[i,'location'].split(',') for i in range(100)])'''



airport_location_2_column=[]
airport_location_2_column=[pd_airport_location.loc[i,'location'].split(',') for i in range(100)]


'''20200909
   需要将list中的str元素转为float,可以了'''
data = list(map(eval, airport_location_2_column[1]))
pd_data=pd.DataFrame(data)





# airport_location.split(",") #字符串转为列表有split(",")
s1 = ','.join(str(n) for n in airport_location)
a=1