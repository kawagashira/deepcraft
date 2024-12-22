#!/usr/bin/env python
#
#                           explore.py
#

import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt
import statsmodels.api as sm

def summarize(i_file):

    df = pd.read_csv(i_file)
    ### 出来高、変化率をFloatに変換 ###
    df = convert_to_float(df)

    ### 欠損値の確認 ###
    print(df.isnull().sum(axis=0))

    df['日付け'] = pd.to_datetime(df['日付け'])
    df['ym_dt'] = list(map(lambda dt: datetime.date(dt.year, dt.month, 1), df['日付け']))
    ym_df = df.groupby('ym_dt').mean()
    print(ym_df)
    show_trend_season(ym_df['始値'])

    print(df)
    print(df.dtypes)
    print(df.describe())


def show_trend_season(ser):

    def plot_aux(i, ind, val, title):

        plt.subplot(4,1,i)
        plt.plot(ind, val)
        plt.ylabel(title)

    plt.figure(figsize=(8,5))
    ser.index = pd.to_datetime(ser.index)
    print(ser)
    res = sm.tsa.seasonal_decompose(ser)
    plot_aux(1, ser.index, ser.values, 'ORIGINAL')
    plt.show()


def convert_to_float(df):

    ### 出来高 ###
    vol = np.array(list(map(to_volume, df['出来高'])))
    print('vol', vol)
    print('vol', type(vol))
    df['出来高'] = vol

    ### 変化率 ###
    change = np.array(list(map(percent2value, df['変化率 %'])))
    print('change', type(change))
    df['変化率 %'] = change
    return df


def to_volume(s):

    if s[-1:] == 'M':       # Million = 10**6
        return float(s[:-1])
    elif s[-1:] == 'B':     # Billion = 10**9
        return float(s[:-1]) * 1000
    else:
        print('ERROR', s)
        return None


def percent2value(s):

    if s[-1:] == '%':
        return float(s[:-1])
    else:
        return None


if __name__ == '__main__':

    i_file = '../assignment-main/Trainee/time-series-prediction/stock_price.csv'
    summarize(i_file)
