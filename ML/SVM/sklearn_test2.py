import pandas as pd
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

breast_data = datasets.load_breast_cancer()
data = pd.DataFrame(datasets.load_breast_cancer().data)

data.columns = breast_data['feature_names']

data_np = breast_data['data']
target_np = breast_data['target']

train_X,test_X, train_y, test_y = train_test_split(data_np,target_np,test_size = 0.1,random_state = 0)

'''
  采用线性核函数进行分类
  kernel可用参数：
  "linear": 线性核函数
  "poly":   多项式核函数
  "rbf" :   径像核函数/高斯核函数
  "sigmoid":核矩阵
'''
model = svm.SVC(kernel='linear', C=2.0)
model.fit(train_X, train_y)

y_pred = model.predict(test_X)
print(accuracy_score(test_y, y_pred))