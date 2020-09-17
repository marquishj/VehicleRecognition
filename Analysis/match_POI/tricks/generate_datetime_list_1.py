import numpy as np
import pandas as pd


def fun(x):
    for i in range(1,11):
        if x.values[i]<10:
            '''将dataframe转为str，并拼接'''
            # date='0'+str(x.values[i][0].tolist())
            date='0'+str(x.values[i].tolist())

    # date=pd.DataFrame(date)
    return date


date=[]
datetime_lhs='2019-09-'
# datetime_rhs=list(map(lambda x: x+1,[2,3]))
datetime_rhs=list(map(lambda x: x,[i for i in range(30)]))
datetime_rhs=pd.DataFrame(datetime_rhs)

datetime_rhs=datetime_rhs.apply(fun)


a=1
