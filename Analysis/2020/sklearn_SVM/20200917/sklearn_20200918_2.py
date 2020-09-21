from sklearn import datasets
iris = datasets.load_iris()
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os

import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import pydotplus

'''https://blog.csdn.net/zxs490862612/article/details/97007536'''

# from sklearn.model_selection import train_test_split
#
# clf = DecisionTreeClassifier()

'''统计日均里程数据'''


# def Daily_data(fileName, Num_train, Num_test, type_vehicle):
#     data = pd.read_excel(fileName, sheet_name=0)  # data_didi_daily_distance
#     data = np.array(data.values[:, 1:]).T




train_daily_distance_mean = []
train_daily_distance_std = []

test_daily_distance_mean = []
test_daily_distance_std = []
target_train = []
target_test = []
# Daily_data(fileName1, 100, 50, 0)  # didi
# Daily_data(fileName2, 130, 70, 1)  # pcev
# Daily_data(fileName3, 70, 30, 1)  # pchev
# Daily_data(fileName4, 100, 50, 0)  # taxi

'''统计周末占比数据'''
weekend_percent_train = []
weekend_percent_test = []

data = pd.read_excel("F:\\sql data\\classifer_car_data\\train data\\5code_didi.xls", sheet_name="所有车周末里程占比")
data = np.array(data.values[:, 1:])  # .fillna(0)
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

data_7 = np.array(pd.read_excel("F:\\sql data\\classifer_car_data\\train data\\6code_时段占比分析.xlsx",
                                sheet_name="所有车型时段7-9.5占比分析").values[:, 1:])
data_16 = np.array(pd.read_excel("F:\\sql data\\classifer_car_data\\train data\\6code_时段占比分析.xlsx",
                                 sheet_name="所有车型时段16.5-18.5占比分析").values[:, 1:])
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
all_data_train.append(train_daily_distance_mean)  #日均里程
all_data_train.append(train_daily_distance_std)   #日均里程方差
all_data_train.append(weekend_percent_train)      #周末占比
all_data_train.append(am_peak_percent_train)      #早高峰出行占比
all_data_train.append(pm_peak_percent_train)      #晚高峰出行占比

all_data_test.append(test_daily_distance_mean)
all_data_test.append(test_daily_distance_std)
all_data_test.append(weekend_percent_test)
all_data_test.append(am_peak_percent_test)
all_data_test.append(pm_peak_percent_test)

all_data_train = np.array(all_data_train).T
all_data_test = np.array(all_data_test).T
'''target_train需要转换格式'''

# target_train = np.array(target_train).reshape(-1, 1)
# clf.fit(all_data_train, target_train)  # 二维数据和一

# 划分

# np.hstack((a,b))  #沿着矩阵列拼接
'''特征'''
vehicle_feature=np.vstack((all_data_test,all_data_train))
np_target_test=np.array(target_test)
np_target_train=np.array(target_train)
'''target_train需要转换格式'''

'''分类标签'''
vehicle_label=np.hstack((np_target_test,np_target_train))

# X_train, X_test, Y_train, Y_test = train_test_split(iris_feature, iris_label, test_size=0.3, random_state=42)
'''需要用混淆矩阵随机打散'''
# X_train, X_test, Y_train, Y_test = train_test_split(iris_feature, iris_label, test_size=0.3, random_state=42)
# x_train, x_test, y_train, y_test = train_test_split(vehicle_feature, vehicle_label, test_size=0.3, random_state=42)

'''增加一维POI匹配的特征向量'''
'''随机森林和Adaboost XGboost'''


def dataProcess():
    # df_didi = pd.read_excel("F:\\sql data\\classifer_car_data\\train data\\5code_didi.xls",'didi').loc[:].values
    df_didi = pd.read_excel("F:\\sql data\\classifer_car_data\\train data\\5code_didi.xls",'didi')
    df_pcev =pd.read_excel( "F:\\sql data\\classifer_car_data\\train data\\5code_pcev.xls",'pcev')
    df_pchev = pd.read_excel("F:\\sql data\\classifer_car_data\\train data\\5code_pchev.xls",'pchev')
    df_taxi =pd.read_excel( "F:\\sql data\\classifer_car_data\\train data\\5code_taxi.xls",'taxi')

    daily_distance_mean_didi=[]
    daily_distance_mean_pcev=[]
    daily_distance_mean_pchev=[]
    daily_distance_mean_taxi=[]
    # daily_distance_mean=map((df_didi.loc[lambda x:x,1]) for i in range(10))
    # df_didi.loc[lambda x:[i for i in range(1,2)]]
    daily_distance_didi=df_didi.iloc[:,lambda x:[i for i in range(1,df_didi.shape[1])]]
    daily_distance_pcev=df_pcev.iloc[:,lambda x:[i for i in range(1,df_pcev.shape[1])]]
    daily_distance_pchev=df_pchev.iloc[:,lambda x:[i for i in range(1,df_pchev.shape[1])]]
    daily_distance_taxi=df_taxi.iloc[:,lambda x:[i for i in range(1,df_taxi.shape[1])]]

    # daily_distance_mean=daily_distance_mean.iloc[:,i [for i in range(10)]].dropna(axis=0, how='any').values
    # daily_distance_mean.iloc[:, lambda x: [i for i in range(10)]].dropna(axis=1, how='any').values
    # for i in range(daily_distance_didi.shape[1]):
    [daily_distance_mean_didi.append(np.mean(daily_distance_didi.iloc[:,i])) for i in range(daily_distance_didi.shape[1])]
    [daily_distance_mean_pcev.append(np.mean(daily_distance_pcev.iloc[:,i])) for i in range(daily_distance_pcev.shape[1])]
    [daily_distance_mean_pchev.append(np.mean(daily_distance_pchev.iloc[:,i])) for i in range(daily_distance_pchev.shape[1])]
    [daily_distance_mean_taxi.append(np.mean(daily_distance_taxi.iloc[:,i])) for i in range(daily_distance_taxi.shape[1])]


    return daily_distance_mean_didi, daily_distance_mean_pcev, daily_distance_mean_pchev, daily_distance_mean_taxi
    # daily_distance_mean=df_didi.loc[i for i in range(10)].apply(lambda x,for i in range(10))


    # for i in range(Num_train):
    #     one_day_data = pd.Series(data[i]).dropna().values  # 删除nan
    #     # one_day_data = one_day_data[one_day_data.apply(lambda x:x>0)].values#.values
    #     train_daily_distance_mean.append(np.mean(one_day_data))
    #     train_daily_distance_std.append(np.std(one_day_data))
    #     target_train.append(type_vehicle)
    # for i in range(Num_train, Num_train + Num_test):
    #     one_day_data = pd.Series(data[i]).dropna().values  # 删除nan
    #     # one_day_data = one_day_data[one_day_data.apply(lambda x:x>0)].values#.values
    #     test_daily_distance_mean.append(np.mean(one_day_data))
    #     test_daily_distance_std.append(np.std(one_day_data))
    #     target_test.append(type_vehicle)


    pass

if __name__ == '__main__':
    daily_distance_mean_didi, daily_distance_mean_pcev, daily_distance_mean_pchev, daily_distance_mean_taxi=dataProcess()

# '''SVM'''
# from sklearn import svm
# svm_classifier = svm.SVC(C=1.0, kernel='rbf', decision_function_shape='ovr', gamma=0.01)
# # svm_classifier.fit(X_train, Y_train)
# svm_classifier.fit(all_data_train, target_train)
#
# print("SVM-训练集:", svm_classifier.score(x_train, y_train))
# # print("训练集:", svm_classifier.score(all_data_train, target_train))
# print("SVM-测试集:", svm_classifier.score(x_test, y_test))
# # print("测试集:", svm_classifier.score(all_data_test, target_test))
#
#
# '''-----------------------XGboost-----------------------'''
# import xgboost
# from xgboost import XGBClassifier
# from numpy import loadtxt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
#
#
# # dataset=loadtxt('F:\PycharmProjects\ACM\(Bili)XGboost\data\pima-indians-diabetes.csv',delimiter=',')
# #
# # X=dataset[:,0:8]
# # Y=dataset[:,8]
#
# seed=7
# test_size=0.33
#
# # x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=test_size,random_state=seed)
#
# model=XGBClassifier()
# model.fit(x_train,y_train)
#
# y_pred=model.predict(x_test)
# predictions=[round(value) for value in y_pred]
#
# accuracy=accuracy_score(y_test,predictions)
# print('XGboost Accuracy:%.2f%%' % (accuracy*100.0))
#
#
# '''-----------------------KNN-----------------------'''
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.datasets import fetch_20newsgroups
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.tree import DecisionTreeClassifier, export_graphviz
#
#
#
# """
# 用KNN算法对鸢尾花进行分类
# :return:
# """
# # 1）获取数据
#
#
# # 2）划分数据集
# # x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)
#
# # 3）特征工程：标准化
# transfer = StandardScaler()
# x_train = transfer.fit_transform(x_train)
# x_test = transfer.transform(x_test)
# print("特征工程：x_train:\n", x_train)
# print("特征工程：x_test:\n", x_test)
#
#
# # 4）KNN算法预估器
# estimator = KNeighborsClassifier(n_neighbors=3)
# estimator.fit(x_train, y_train)
#
# # 5）模型评估
# # 方法1：直接比对真实值和预测值
# y_predict = estimator.predict(x_test)
# print("y_predict:\n", y_predict)
# print("直接比对真实值和预测值:\n", y_test == y_predict)
#
# # 方法2：计算准确率
# score = estimator.score(x_test, y_test)
# print("KNN准确率为：\n", score)
#
#
# '''-----------------KNN++----------------'''
# # 3）特征工程：标准化
# # transfer = StandardScaler()
# # x_train = transfer.fit_transform(x_train)
# # x_test = transfer.transform(x_test)
# #
# # # 4）KNN算法预估器
# # estimator = KNeighborsClassifier()
# #
# # # 加入网格搜索与交叉验证
# # # 参数准备
# # param_dict = {"n_neighbors": [1, 3, 5, 7, 9, 11]}
# # estimator = GridSearchCV(estimator, param_grid=param_dict, cv=10)
# # estimator.fit(x_train, y_train)
# #
# # # 5）模型评估
# # # 方法1：直接比对真实值和预测值
# # y_predict = estimator.predict(x_test)
# # print("y_predict:\n", y_predict)
# # print("直接比对真实值和预测值:\n", y_test == y_predict)
# #
# # # 方法2：计算准确率
# # score = estimator.score(x_test, y_test)
# # print("准确率为：\n", score)
# #
# # # 最佳参数：best_params_
# # print("最佳参数：\n", estimator.best_params_)
# # # 最佳结果：best_score_
# # print("最佳结果：\n", estimator.best_score_)
# # # 最佳估计器：best_estimator_
# # print("最佳估计器:\n", estimator.best_estimator_)
# # # 交叉验证结果：cv_results_
# # print("交叉验证结果:\n", estimator.cv_results_)
#
# '''Random Forest'''
# '''https://zhuanlan.zhihu.com/p/126288078'''
#
#
# '''----------------Naive Bayers----------------'''
#
#
# # 4）朴素贝叶斯算法预估器流程
# # estimator = MultinomialNB()
# # estimator.fit(x_train, y_train)
# #
# # # 5）模型评估
# # # 方法1：直接比对真实值和预测值
# # y_predict = estimator.predict(x_test)
# # print("y_predict:\n", y_predict)
# # print("直接比对真实值和预测值:\n", y_test == y_predict)
# #
# # # 方法2：计算准确率
# # score = estimator.score(x_test, y_test)
# # print("Naive Bayers准确率为：\n", score)
#
#
# '''--------------决策树-----------------'''
# estimator = DecisionTreeClassifier(criterion="entropy")
# estimator.fit(x_train, y_train)
#
# # 4）模型评估
# # 方法1：直接比对真实值和预测值
# y_predict = estimator.predict(x_test)
# print("y_predict:\n", y_predict)
# print("直接比对真实值和预测值:\n", y_test == y_predict)
#
# # 方法2：计算准确率
# score = estimator.score(x_test, y_test)
# print("Decision Tree 准确率为：\n", score)

# 可视化决策树
# export_graphviz(estimator, out_file="iris_tree.dot", feature_names=iris.feature_names)