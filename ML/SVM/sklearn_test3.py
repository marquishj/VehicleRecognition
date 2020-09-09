from sklearn import datasets
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

'''鸢尾花作为数据集练手机器学习
https://www.jianshu.com/p/e6dd565796e5'''

iris = datasets.load_iris()
index=np.array(['Sepal_Length', 'Sepal_Width','Petal_Length','Petal_Width'])
df=pd.DataFrame(iris.data,columns=index)
df.describe()



# %matplotlib inline

#按鸢尾花的标签涂色
def scatter_plot_by_category(feat, x, y):
    alpha = 0.5
    gs = df.groupby(feat)
    cs = cm.rainbow(np.linspace(0, 1, len(gs)))
    for g, c in zip(gs, cs):
        plt.scatter(g[1][x], g[1][y], color=c, alpha=alpha)

plt.figure(figsize=(20,5))

plt.subplot(131)
scatter_plot_by_category('Species', 'Sepal_Length', 'Petal_Length')
plt.xlabel('Sepal_Length')
plt.ylabel('Petal_Length')
plt.title('Species')