# -*- coding: utf-8 -*-
# @Time    : 2020/10/27 16:49
# @Author  : Hou Jue
# @Email   : 382062104@qq.com
# @File    : all_combination_3.py
# @Software: PyCharm
# -*- coding: utf-8 -*-
import numpy as np
# -*- coding: utf-8 -*-

def sub_sets(a,b,idx):
    if(idx==len(a)):
        print(b)

    else:
        c=b[:]
        b.append(a[idx])
        sub_sets(a,b,idx+1)
        sub_sets(a,c,idx+1)



if __name__ == '__main__':
    '''BEV特征提取'''
    BEV_daily_distance_mean = [0,0,0,0,0,0,0,0]  # 统计日均里程
    BEV_daily_distance_std = [1,1,1,1,1,1,1,1]  # 统计日里程标准差
    BEV_night_distance_mean = [2,2,2,2,2,2,2,2]  ##统计夜间日均里程
    BEV_night_distance_percentage = [3,3,3,3,3,3,3,3]  # 统计夜间里程占比 0：00-5：00
    BEV_am_peak_percentage = [4,4,4,4,4,4,4,4]  # 统计早高峰里程占比  7：00-9：00
    BEV_pm_peak_percentage = [5,5,5,5,5,5,5,5]  # 统计晚高峰里程占比  17：00-19：00
    BEV_weekends_distance_percentage = [6,6,6,6,6,6,6,6]  # 统计周末里程占比
    BEV_charging_rate_mean = [7,7,7,7,7,7,7,7]  # 统计充电速率


    X = np.array([BEV_daily_distance_mean, \
                  BEV_daily_distance_std, \
                  BEV_night_distance_mean, \
                  BEV_night_distance_percentage, \
                  BEV_am_peak_percentage, \
                  BEV_pm_peak_percentage, \
                  BEV_weekends_distance_percentage, \
                  BEV_charging_rate_mean])

    XX = {0: BEV_daily_distance_mean,
          1: BEV_daily_distance_std,
          2: BEV_night_distance_mean,
          3: BEV_night_distance_percentage,
          4: BEV_am_peak_percentage,
          5: BEV_pm_peak_percentage,
          6: BEV_weekends_distance_percentage,
          7: BEV_charging_rate_mean,
          }

    X_key = [XX[i] for i in range(8)]
    # sub_sets([1,2,3],[],0)
    sub_sets(X_key,[],0)