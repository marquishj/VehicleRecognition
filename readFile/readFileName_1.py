import os
# 保存不同模型的目录名(绝对路径)
file_dir = r'F:\sql data\Step5_data_process_5_sql_20200708'
bayes_list = []     # 贝叶斯模型列表
svm_list = []        # svm模型列表
xgboost_list = []  # xgboost模型列表
# root是指当前目录路径(文件夹的绝对路径)
# dirs是指路径下所有的子目录(文件夹里的文件夹)
# files是指路径下所有的文件(文件夹里所有的文件)
for root,dirs,files in os.walk(file_dir):
    for file in files:
        if os.path.splitext(file)[0] == 'bayes':
            bayes_list.append(os.path.join(root,file))
        elif os.path.splitext(file)[0] == 'svm':
            svm_list.append(os.path.join(root,file))
        elif os.path.splitext(file)[0] == 'xgboost':
            xgboost_list.append(os.path.join(root,file))