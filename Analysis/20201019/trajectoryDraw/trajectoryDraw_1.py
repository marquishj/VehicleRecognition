# -*- coding: utf-8 -*-
# @Time    : 2020/10/19 15:18
# @Author  : Hou Jue
# @Email   : 382062104@qq.com
# @File    : trajectoryDraw_1.py
# @Software: PyCharm

# Python画点或者曲线的轨迹，并保存为动图。
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
# Inputfile = np.loadtxt('SJ.txt')  # 滞回曲线存在SJ.txt里
# Inputfile=np.loadtxt('F:\sql data\classifer_car_data\origin\data.txt')

# Inputfile=np.loadtxt('F:\sql data\classifer_car_data\example\SHEVDC_0A101F56_vehicle_position.csv',delimiter=",")
# p = r'F:\sql data\classifer_car_data\example\SHEVDC_0A101F56_vehicle_position.csv'



Inputfile=pd.read_csv('F:\sql data\classifer_car_data\example\SHEVDC_0A101F56_vehicle_position.csv')


# with open(p,encoding = 'utf-8') as f:
#     data = np.loadtxt(f,str,delimiter = ",")
#     print(data[:5])


lat=Inputfile['latitude']
lon=Inputfile['longitude']
#
Inputfile=pd.concat([lat,lon],axis=1)
Inputfile=Inputfile.to_numpy()
rows=Inputfile.shape[0]

if __name__ == "__main__":
    fig = plt.figure()
    xx = []
    yy = []
    yyyy=[]
    for i in range(rows - 1):
        x = Inputfile[i, 0]
        xx.append(x)
        y = Inputfile[i, 1]
        yy.append(y)
        yyy=plt.plot(xx, yy,'b')
        yyyy.append(yyy)
    # ani = animation.ArtistAnimation(fig, yyyy, interval=0.001, repeat_delay=1000)
    # ani = animation.ArtistAnimation(fig, yyyy, interval=1,blit=True)
    ani = animation.ArtistAnimation(fig, yyyy, interval=1,blit=False)
    # ani.save("F:\\sql data\\classifer_car_data\\origin\\test3.gif",writer='pillow')
    ani.save("F:\\sql data\\classifer_car_data\\origin\\test10.gif",writer='pillow')
    print("finish")