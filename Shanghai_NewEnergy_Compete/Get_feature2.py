import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import datetime

'''异常值处理函数'''
# 重复数据清洗
def DuplicateData_cleaning(df):
    data=df.groupby('datatime')['summileage'].sum().reset_index()
    return data
# 异常数据清洗
def Outlier_cleaning(data):
    licheng = data['summileage']
    LC = list(licheng)
    A = []            # 异常值所在位置
    B = []          # 正常值所在位置
    for i in range(len(LC)):
        if LC[i] > 999999:
            A.append(i)
        else:
            B.append(i)
    ZC = []
    for m in range(len(B)):
        zc = LC[B[m]]
        ZC.append(zc)
    if A != []:
        if A[0] == 0:
            s = 0
            for q in range(1, len(A) ):
                if A[q] - A[q - 1] == 1:
                    s = q + 1 
                else:
                    break
            f=interp1d(B, ZC, kind = 'slinear')       # 插值
            fnew = f(A[s:])
            for n in range(len(A[s:])):
                LC[A[s + n]] = fnew[n]
            B.extend(A[s :])   # 线性拟合
            ZC.extend(fnew)
            z1 = np.polyfit(B, ZC, 1)
            p1 = np.poly1d(z1)
            yvals = p1(A[0: s+1])
            a =  yvals[-1] - ZC[0]

            for p in range(s):
                LC[p] = yvals[p] - a

            # plt.plot(range(len(LC)), LC)
            # plt.show
        else:
            f=interp1d(B, ZC, kind = 'slinear')
            fnew = f(A)
            for nn in range(len(A)):
                LC[A[nn]] = fnew[nn]
            # plt.plot(range(len(LC)), LC)
            # plt.show

    for jj in range(len(LC)):  #判断负值
        if LC[jj] == 0:
            print(BEV_id)
            break
    data['summileage'] = LC
    return data

'''计算里程'''
def find_miles(mileage):
    mileage=list(mileage)
    origion = mileage[0]
    destination = mileage[-1]
    distance = destination - origion
    return distance

# 计算充电速率函数
def Charging_rate(v1):
    v1 = v1[v1['chargestatus'] != 254] #筛除异常值
    A = v1[v1['chargestatus'] == 1]  #获取停车充电数据的索引
    if len(A) != 0:
        charging_index = A.index        # 充电状态为1的index
        STOP = []
        START =[charging_index[0]]
        for i in range(1, len(charging_index) ):
            if charging_index[i] - charging_index[i - 1] != 1:        # index不连续
                stop_charge = charging_index[i - 1]                   # 最后一个1为停止充电的点
                STOP.append(stop_charge)
                START.append(charging_index[i])
        if (len(START) - len(STOP) == 1):                       # 补齐结尾
            STOP.append(charging_index[-1])
        SPEED = []  #每一段充电的速度
        charging_times_list = [] #每一段充电的秒数
        SOC_dif = []  # 每一段充电量SOC差值
        for j in range(len(STOP)):
            if START[j] != STOP[j]:
                df = v1[START[j] : STOP[j]]
                #计算每一段充电时间
                ks = datetime.datetime.strptime(df['datatime'].iloc[0], '%Y-%m-%d %H:%M:%S')  #每一段充电开始时间
                js = datetime.datetime.strptime(df['datatime'].iloc[-1], '%Y-%m-%d %H:%M:%S')  ##每一段充电结束时间
                charging_time = (js - ks).seconds  
                charging_times_list.append(charging_time)
                #计算每一段充电量即SOC差值
                ks = df['soc'].iloc[0]
                js = df['soc'].iloc[-1]
                dif = js - ks 
                SOC_dif.append(dif)
        SPEED = np.array(SOC_dif)/np.array(charging_times_list)
        SPEED_mean = np.nanmean(SPEED)
    else:
        SPEED_mean = nan
    return SPEED_mean


'''获取周末的日期'''
weekends_6 = datetime.datetime.strptime('2019-01-05', '%Y-%m-%d')
weekends_7 = datetime.datetime.strptime('2019-01-06', '%Y-%m-%d')
delta = datetime.timedelta(days = 7)
W6 = [weekends_6]
W7 = [weekends_7]
for i in range(25):
    w6 = weekends_6 + (i + 1)* delta
    w7 = weekends_7 + (i + 1)* delta
    W6.append(w6)
    W7.append(w7)
weekend = W6 + W7
weekend=[x.date() for x in weekend]
# print(weekend)
# print(W6)
# print(W7)


BEV_list = os.listdir("/home/kesci/input/data2737/BEV")
PHEV_list = os.listdir("/home/kesci/input/data2737/PHEV")


'''BEV特征提取'''
BEV_daily_distance_mean =[]  #统计日均里程
BEV_daily_distance_std = []  #统计日里程标准差
BEV_night_distance_mean = [] ##统计夜间日均里程
BEV_night_distance_percentage = [] #统计夜间里程占比 0：00-5：00
BEV_am_peak_percentage = [] #统计早高峰里程占比  7：00-9：00
BEV_pm_peak_percentage = []  #统计晚高峰里程占比  17：00-19：00
BEV_weekends_distance_percentage = [] #统计周末里程占比 
BEV_charging_rate_mean =[]  #统计充电速率

# Num_BEV = len(BEV_list)
# i = 0
# # IDs = []
# for BEV_id in BEV_list:
#     i += 1
#     print(i,BEV_id)
#     # IDs.append(BEV_id[:-4])
#     if i <=Num_BEV:
#         v1=pd.read_csv("/home/kesci/input/data2737/BEV/"+BEV_id)
#         # v1 = DuplicateData_cleaning(v1)
#         v1 = Outlier_cleaning(v1)

#         v1['day']=pd.to_datetime(v1['datatime']).apply(lambda x:x.date())  #year,month,day
#         '''统计日均里程与日里程标准差'''
#         day_distance = v1.groupby('day').agg({'summileage':find_miles})
#         # print(sum(list(day_distance["summileage"])))
#         # BEV_daily_distance_mean.append(np.mean(list(day_distance["summileage"])))
#         # BEV_daily_distance_std.append(np.std(list(day_distance["summileage"])))
        
#         '''统计周末里程占比'''
#         weekend_data = v1[v1['day'].isin(weekend)]
#         weekend_distance = weekend_data.groupby('day').agg({'summileage':find_miles})
#         weekend_percent = sum(list(weekend_distance["summileage"]))/sum(list(day_distance["summileage"]))
#         BEV_weekends_distance_percentage.append(weekend_percent)



#         v1['hour'] = pd.to_datetime(v1['datatime']).apply(lambda x:x.strftime('%X'))
#         '''统计夜间日均里程'''
#         night_data = v1[(pd.to_datetime(v1['hour'],format = '%H:%M:%S') >=pd.to_datetime('00:00:00',format = '%H:%M:%S')) & (pd.to_datetime(v1['hour'],format = '%H:%M:%S') <= pd.to_datetime('05:00:00',format = '%H:%M:%S'))]
#         night_mile = night_data.groupby('day').agg({'summileage':find_miles})
#         BEV_night_distance_mean.append(np.mean(list(night_mile["summileage"])))

#         '''统计夜间里程占比'''
#         night_percent = sum(list(night_mile["summileage"]))/sum(list(day_distance["summileage"]))
#         BEV_night_distance_percentage.append(night_percent)

#         '''统计早高峰里程占比'''
#         am_peak_data = v1[(pd.to_datetime(v1['hour'],format = '%H:%M:%S') >=pd.to_datetime('07:00:00',format = '%H:%M:%S')) & (pd.to_datetime(v1['hour'],format = '%H:%M:%S') <= pd.to_datetime('09:00:00',format = '%H:%M:%S'))]
#         am_peak_mile = am_peak_data.groupby('day').agg({'summileage':find_miles})
#         am_peak_percent = sum(list(am_peak_mile["summileage"]))/sum(list(day_distance["summileage"]))
#         BEV_am_peak_percentage.append(am_peak_percent)
        
#         '''统计晚高峰里程占比'''
#         pm_peak_data = v1[(pd.to_datetime(v1['hour'],format = '%H:%M:%S') >=pd.to_datetime('17:00:00',format = '%H:%M:%S')) & (pd.to_datetime(v1['hour'],format = '%H:%M:%S') <= pd.to_datetime('19:00:00',format = '%H:%M:%S'))]
#         pm_peak_mile = pm_peak_data.groupby('day').agg({'summileage':find_miles})
#         pm_peak_percent = sum(list(pm_peak_mile["summileage"]))/sum(list(day_distance["summileage"]))
#         BEV_pm_peak_percentage.append(pm_peak_percent)

#         charging_speed = Charging_rate(v1)
#         BEV_charging_rate_mean.append(charging_speed)



'''PHEV特征提取'''
PHEV_daily_distance_mean =[]  #统计日均里程
PHEV_daily_distance_std = []  #统计日里程标准差
PHEV_night_distance_mean = [] ##统计夜间日均里程
PHEV_night_distance_percentage = [] #统计夜间里程占比 0：00-5：00
PHEV_am_peak_percentage = [] #统计早高峰里程占比  7：00-9：00
PHEV_pm_peak_percentage = []  #统计晚高峰里程占比  17：00-19：00
PHEV_weekends_distance_percentage = [] #统计周末里程占比 
PHEV_charging_rate_mean =[]  #统计充电速率

Num_PHEV = len(PHEV_list)
i = 0
# IDs = []
for PHEV_id in PHEV_list:
    i += 1
    print(i,PHEV_id)
    # IDs.append(BEV_id[:-4])
    if i <=Num_PHEV:
        v1=pd.read_csv("/home/kesci/input/data2737/PHEV/"+PHEV_id)
        #v1 = DuplicateData_cleaning(v1)
        v1 = Outlier_cleaning(v1)
        # v1['day']=pd.to_datetime(v1['datatime']).apply(lambda x:x.date())  #year,month,day
        
        '''统计日均里程与日里程标准差'''
        day_distance = v1.groupby('day').agg({'summileage':find_miles})
        PHEV_daily_distance_mean.append(np.mean(list(day_distance["summileage"])))
        PHEV_daily_distance_std.append(np.std(list(day_distance["summileage"])))
        
        # '''统计周末里程占比'''
        # weekend_data = v1[v1['day'].isin(weekend)]
        # weekend_distance = weekend_data.groupby('day').agg({'summileage':find_miles})
        # weekend_percent = sum(list(weekend_distance["summileage"]))/sum(list(day_distance["summileage"]))
        # PHEV_weekends_distance_percentage.append(weekend_percent)


        # v1['hour'] = pd.to_datetime(v1['datatime']).apply(lambda x:x.strftime('%X'))
        # # '''统计夜间日均里程'''
        # night_data = v1[(pd.to_datetime(v1['hour'],format = '%H:%M:%S') >=pd.to_datetime('00:00:00',format = '%H:%M:%S')) & (pd.to_datetime(v1['hour'],format = '%H:%M:%S') <= pd.to_datetime('05:00:00',format = '%H:%M:%S'))]
        # night_mile = night_data.groupby('day').agg({'summileage':find_miles})
        # PHEV_night_distance_mean.append(np.mean(list(night_mile["summileage"])))

        # # '''统计夜间里程占比'''
        # night_percent = sum(list(night_mile["summileage"]))/sum(list(day_distance["summileage"]))
        # PHEV_night_distance_percentage.append(night_percent)

        # '''统计早高峰里程占比'''
        # am_peak_data = v1[(pd.to_datetime(v1['hour'],format = '%H:%M:%S') >=pd.to_datetime('07:00:00',format = '%H:%M:%S')) & (pd.to_datetime(v1['hour'],format = '%H:%M:%S') <= pd.to_datetime('09:00:00',format = '%H:%M:%S'))]
        # am_peak_mile = am_peak_data.groupby('day').agg({'summileage':find_miles})
        # am_peak_percent = sum(list(am_peak_mile["summileage"]))/sum(list(day_distance["summileage"]))
        # PHEV_am_peak_percentage.append(am_peak_percent)
        
        # '''统计晚高峰里程占比'''
        # pm_peak_data = v1[(pd.to_datetime(v1['hour'],format = '%H:%M:%S') >=pd.to_datetime('17:00:00',format = '%H:%M:%S')) & (pd.to_datetime(v1['hour'],format = '%H:%M:%S') <= pd.to_datetime('19:00:00',format = '%H:%M:%S'))]
        # pm_peak_mile = pm_peak_data.groupby('day').agg({'summileage':find_miles})
        # pm_peak_percent = sum(list(pm_peak_mile["summileage"]))/sum(list(day_distance["summileage"]))
        # PHEV_pm_peak_percentage.append(pm_peak_percent)

        charging_speed = Charging_rate(v1)
        PHEV_charging_rate_mean.append(charging_speed)

print("over")

