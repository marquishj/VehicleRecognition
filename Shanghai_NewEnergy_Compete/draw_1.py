# -*- coding: utf-8 -*-
# @Time    : 2020/10/30 14:51
# @Author  : Hou Jue
# @Email   : 382062104@qq.com
# @File    : draw_1.py
# @Software: PyCharm
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.metrics import silhouette_score # 导入轮廓系数指标
from sklearn.cluster import KMeans # KMeans模块
# %matplotlib inline
# import pandas as pd
model_scaler = MinMaxScaler()  # 建立MinMaxScaler模型对象
num_sets=[[933.015,0.003,0.064,5.916,0.006,8.770],
        [1390.013,0.003,0.152,1.168,0.017,8.199],
        [2717.419,0.005,0.051,0.947,0.007,8.529],
        [1904.371,0.003,0.106,0.943,0.009,8.217]]
num_sets=pd.DataFrame(num_sets)
num_sets_max_min = model_scaler.fit_transform(num_sets)

# 画图
fig = plt.figure(figsize=(6,6))  # 建立画布
ax = fig.add_subplot(111, polar=True)  # 增加子网格，注意polar参数
# labels = np.array(merge_data1.index)  # 设置要展示的数据标签
labels = np.array(['1','2','3','4','5'])  # 设置要展示的数据标签
cor_list = ['g', 'r', 'y', 'b']  # 定义不同类别的颜色
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)  # 计算各个区间的角度
angles = np.concatenate((angles, [angles[0]]))  # 建立相同首尾字段以便于闭合
# 画雷达图
for i in range(len(num_sets)):  # 循环每个类别
    data_tmp = num_sets_max_min[i, :]  # 获得对应类数据
    data = np.concatenate((data_tmp, [data_tmp[0]]))  # 建立相同首尾字段以便于闭合
    ax.plot(data, 'o-', c=cor_list[i], label="第%d类渠道"%(i))  # 画线
    # ax.plot(angles, data, 'o-', c=cor_list[i], label="第%d类渠道"%(i))  # 画线
    ax.fill( data,alpha=2.5)
    # ax.fill(angles, data,alpha=2.5)
# 设置图像显示格式
ax.set_thetagrids(angles * 180 / np.pi, labels, fontproperties="SimHei")  # 设置极坐标轴
ax.set_title("各聚类类别显著特征对比", fontproperties="SimHei")  # 设置标题放置
ax.set_rlim(-0.2, 1.2)  # 设置坐标轴尺度范围
plt.legend(loc="upper right" ,bbox_to_anchor=(1.2,1.0))  # 设置图例位置
plt.show()