# -*- coding: utf-8 -*-
# @Time    : 2020/10/27 16:49
# @Author  : Hou Jue
# @Email   : 382062104@qq.com
# @File    : all_combination_3.py
# @Software: PyCharm
# -*- coding: utf-8 -*-
import numpy as np
# -*- coding: utf-8 -*-





def Combinations(L, k):
    """List all combinations: choose k elements from list L"""
    n = len(L)
    result = []  # To Place Combination result
    for i in range(n - k + 1):
        if k > 1:
            newL = L[i + 1:]
            Comb, _ = Combinations(newL, k - 1)
            for item in Comb:
                item.insert(0, L[i])
                result.append(item)
        else:
            result.append([L[i]])
    return result, len(result)


if __name__ == '__main__':
    '''BEV特征提取'''
    BEV_daily_distance_mean = [0]  # 统计日均里程
    BEV_daily_distance_std = [1]  # 统计日里程标准差
    BEV_night_distance_mean = [2]  ##统计夜间日均里程
    BEV_night_distance_percentage = [3]  # 统计夜间里程占比 0：00-5：00
    BEV_am_peak_percentage = [4]  # 统计早高峰里程占比  7：00-9：00
    BEV_pm_peak_percentage = [5]  # 统计晚高峰里程占比  17：00-19：00
    BEV_weekends_distance_percentage = [6]  # 统计周末里程占比
    BEV_charging_rate_mean = [7]  # 统计充电速率

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
    print(XX)
    print(Combinations(range(5), 3))
    # print(Combinations([0,1,2,3,4], 3))
    X_key=[XX[i] for i in range(8)]
    print(Combinations(X_key, 3))
