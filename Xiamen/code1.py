import pandas as pd
import numpy as np
import glob

'''我们可以很简单计算得出运行结果：
计算巡游车日均空驶率、运距和运行时长；
计算网约车日均空驶率、运距和运行时长；'''

# 网约车计算
def cal_wyc(df):
    df = df[['DEST_TIME', 'DEP_TIME', 'WAIT_MILE', 'DRIVE_MILE']].dropna()

    if df['DEST_TIME'].dtype != np.int64:
        df = df[df['DEST_TIME'].apply(len) == 14]
        df = df[df['DEST_TIME'].apply(lambda x: x.isdigit())]

    df['DEP_TIME'] = pd.to_datetime(df['DEP_TIME'], format='%Y%m%d%H%M%S')
    df['DEST_TIME'] = pd.to_datetime(df['DEST_TIME'], format='%Y%m%d%H%M%S')

    # df = df[df['DRIVE_MILE'].apply(lambda x: '-' not in str(x) and '|' not in str(x) and '路' not in str(x))]
    df['DRIVE_MILE'] = df['DRIVE_MILE'].astype(float)
    df['WAIT_MILE'] = df['WAIT_MILE'].astype(float)

    # return df
    print('空驶率：', (df['WAIT_MILE'] / (df['WAIT_MILE'] + df['DRIVE_MILE'] + 0.01)).mean())
    print('订单平均距离：', df['DRIVE_MILE'].dropna().mean())
    print('订单平均时长：', ((df['DEST_TIME'] - df['DEP_TIME']).dt.seconds / 60.0).mean())

# 巡游车计算
def cal_taxi(df):
    df['GETON_DATE'] = pd.to_datetime(df['GETON_DATE'])
    df['GETOFF_DATE'] = pd.to_datetime(df['GETOFF_DATE'])

    print('空驶率：', (df['NOPASS_MILE'] / (df['NOPASS_MILE'] + df['PASS_MILE'])).mean())
    print('订单平均距离：', df['PASS_MILE'].mean())
    print('订单平均时长：', ((df['GETOFF_DATE'] - df['GETON_DATE']).dt.seconds / 60.0).mean())

'''2019年端午节数据：'''
INPUT_PATH = 'J:/Xiamen/input/'
df = taxiorder2019 = pd.concat([
    pd.read_csv(INPUT_PATH + x) for x in [
        'taxiOrder20190607.csv',
        'taxiOrder20190608.csv',
        'taxiOrder20190609.csv'
    ]
])
cal_taxi(df)


INPUT_PATH = 'J:/Xiamen/input/'
df = taxiorder2019 = pd.concat([
    pd.read_csv(INPUT_PATH + x) for x in [
        'wycOrder20190607.csv',
        'wycOrder20190608.csv',
        'wycOrder20190609.csv'
    ]
])
cal_wyc(df)

'''2019年工作日数据：'''
INPUT_PATH = 'J:/Xiamen/input/'
df = taxiorder2019 = pd.concat([
    pd.read_csv(INPUT_PATH + x) for x in [
        'taxiOrder20190531.csv',
        'taxiOrder20190603.csv',
        'taxiOrder20190604.csv',
        'taxiOrder20190605.csv',
        'taxiOrder20190606.csv'
    ]
])
cal_taxi(df)


INPUT_PATH = 'J:/Xiamen/input/'
df = taxiorder2019 = pd.concat([
    pd.read_csv(INPUT_PATH + x) for x in [
        'wycOrder20190531.csv',
        'wycOrder20190603.csv',
        'wycOrder20190604.csv',
        'wycOrder20190605.csv',
        'wycOrder20190606.csv'
    ]
])
cal_wyc(df)

'''2019年周末数据：'''
INPUT_PATH = 'J:/Xiamen/input/'
df = taxiorder2019 = pd.concat([
    pd.read_csv(INPUT_PATH + x) for x in [
        'taxiOrder20190601.csv',
        'taxiOrder20190602.csv',
    ]
])
cal_taxi(df)

df = taxiorder2019 = pd.concat([
    pd.read_csv(INPUT_PATH + x) for x in [
        'wycOrder20190601.csv',
        'wycOrder20190602.csv',
    ]
])
cal_wyc(df)