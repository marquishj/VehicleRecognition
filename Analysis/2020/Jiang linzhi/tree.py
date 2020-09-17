# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import pydotplus 

clf = DecisionTreeClassifier()

'''统计日均里程数据'''
def Daily_data(fileName,Num_train,Num_test,type_vehicle):
    
    data = pd.read_excel(fileName,sheet_name = 0) #data_didi_daily_distance
    data = np.array(data.values[:,1:]).T  
    for i in range(Num_train):
        one_day_data = pd.Series(data[i]).dropna().values #删除nan
        # one_day_data = one_day_data[one_day_data.apply(lambda x:x>0)].values#.values
        train_daily_distance_mean.append(np.mean(one_day_data))
        train_daily_distance_std.append(np.std(one_day_data))
        target_train.append(type_vehicle)
    for i in range(Num_train,Num_train+Num_test):
        one_day_data = pd.Series(data[i]).dropna().values #删除nan
        # one_day_data = one_day_data[one_day_data.apply(lambda x:x>0)].values#.values
        test_daily_distance_mean.append(np.mean(one_day_data))
        test_daily_distance_std.append(np.std(one_day_data))
        target_test.append(type_vehicle)

fileName1 = "F:\\sql data\\classifer_car_data\\train data\\5code_didi.xls"
fileName2 = "F:\\sql data\\classifer_car_data\\train data\\5code_pcev.xls"
fileName3 = "F:\\sql data\\classifer_car_data\\train data\\5code_pchev.xls"
fileName4 = "F:\\sql data\\classifer_car_data\\train data\\5code_taxi.xls"

train_daily_distance_mean =[]
train_daily_distance_std = []


test_daily_distance_mean =[]
test_daily_distance_std = []
target_train = []
target_test = []
Daily_data(fileName1,100,50,0)  #didi
Daily_data(fileName2,130,70,1)  #pcev
Daily_data(fileName3,70,30,1)   #pchev
Daily_data(fileName4,100,50,0)  #taxi


'''统计周末占比数据'''
weekend_percent_train = []
weekend_percent_test = []

data = pd.read_excel("F:\\sql data\\classifer_car_data\\train data\\5code_didi.xls",sheet_name = "所有车周末里程占比")
data = np.array(data.values[:,1:])  #.fillna(0)
weekend_percent_train.extend(data[0][:100])
weekend_percent_train.extend(data[1][:130])
weekend_percent_train.extend(data[2][:70])
weekend_percent_train.extend(data[3][:100])

weekend_percent_test.extend(data[0][100:150])
weekend_percent_test.extend(data[1][130:200])
weekend_percent_test.extend(data[2][70:100])
weekend_percent_test.extend(data[3][100:150])

'''统计时段占比数据'''
am_peak_percent_train = []
am_peak_percent_test = []
pm_peak_percent_train = []
pm_peak_percent_test = []


data_7 = np.array(pd.read_excel("F:\\sql data\\classifer_car_data\\train data\\6code_时段占比分析.xlsx",sheet_name = "所有车型时段7-9.5占比分析").values[:,1:])
data_16 = np.array(pd.read_excel("F:\\sql data\\classifer_car_data\\train data\\6code_时段占比分析.xlsx",sheet_name = "所有车型时段16.5-18.5占比分析").values[:,1:])
# am
am_peak_percent_train.extend(data_7[0][:100])
am_peak_percent_train.extend(data_7[1][:130])
am_peak_percent_train.extend(data_7[2][:70])
am_peak_percent_train.extend(data_7[3][:100])

am_peak_percent_test.extend(data_7[0][100:150])
am_peak_percent_test.extend(data_7[1][130:200])
am_peak_percent_test.extend(data_7[2][70:100])
am_peak_percent_test.extend(data_7[3][100:150])


# pm
pm_peak_percent_train.extend(data_16[0][:100])
pm_peak_percent_train.extend(data_16[1][:130])
pm_peak_percent_train.extend(data_16[2][:70])
pm_peak_percent_train.extend(data_16[3][:100])

pm_peak_percent_test.extend(data_16[0][100:150])
pm_peak_percent_test.extend(data_16[1][130:200])
pm_peak_percent_test.extend(data_16[2][70:100])
pm_peak_percent_test.extend(data_16[3][100:150])




all_data_train = []
all_data_test = []
all_data_train.append(train_daily_distance_mean)
all_data_train.append(train_daily_distance_std)
all_data_train.append(weekend_percent_train)
all_data_train.append(am_peak_percent_train)
all_data_train.append(pm_peak_percent_train)

all_data_test.append(test_daily_distance_mean)
all_data_test.append(test_daily_distance_std)
all_data_test.append(weekend_percent_test)
all_data_test.append(am_peak_percent_test)
all_data_test.append(pm_peak_percent_test)

all_data_train = np.array(all_data_train).T
all_data_test = np.array(all_data_test).T
target_train = np.array(target_train).reshape(-1, 1)
clf.fit(all_data_train,target_train)  #二维数据和一维数组

target_predict = clf.predict(all_data_test)
print("准确率:",accuracy_score(target_test, target_predict))


dot_data = tree.export_graphviz(clf, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("presict.pdf") 


# =============================================================================
#  与accuracy_score功能一样
# result = (target_predict == target_test)   # True则预测正确，False则预测错误
# True_predict = result[result==True]
# print(len(True_predict)/len(result))
# 
# =============================================================================

# =============================================================================
# from IPython.display import Image  
# import pydotplus
# dot_data = tree.export_graphviz(clf, out_file=None, 
#                           # feature_names=iris.feature_names,  
#                           # class_names=iris.target_names,  
#                           filled=True, rounded=True,  
#                           special_characters=True)  
# graph = pydotplus.graph_from_dot_data(dot_data)  
# Image(graph.create_png()) 
# =============================================================================
