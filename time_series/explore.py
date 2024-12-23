#!/usr/bin/env python
#
#                           explore.py
#

import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt
import statsmodels.api as sm
import os

def summarize(i_file):

    df = pd.read_csv(i_file)
    ### 出来高、変化率をFloatに変換 ###
    df = convert_to_float(df)

    ### 欠損値の確認 ###
    print(df.isnull().sum(axis=0))

    ### 月別データに変換 ###
    df['日付け'] = pd.to_datetime(df['日付け'])
    df['ym_dt'] = list(map(lambda dt: datetime.date(dt.year, dt.month, 1), df['日付け']))
    ym_df = df.groupby('ym_dt').mean()
    print(ym_df)

    ### トレンド、季節性の確認 ###
    o_dir = 'fig'
    if not os.path.isdir(o_dir):
        os.mkdir(o_dir)
    for col in ['始値', '終値', '安値', '高値']: 
        o_file = '%s/decompose-%s.png' % (o_dir, col)
        print('OUTPUT', o_file)
        show_trend_season(ym_df[col], o_file)

    ### 自己相関係数の確認 ###
    o_file = '%s/acf-終値.png' % o_dir
    print('ACF', o_file)
    show_acf(ym_df['終値'], o_file)

    ### 階差系列 ###
    ym_diff = (ym_df - ym_df.shift()).dropna()
    print(ym_diff)
    o_file = '%s/diff-asf-終値.png' % o_dir
    show_trend_season(ym_diff['終値'], o_file)

    ### 階差系列12 ##
    ym_diff12 = (ym_df - ym_df.shift(12)).dropna()
    o_file = '%s/diff12-asf-終値.png' % o_dir
    show_trend_season(ym_diff12['終値'], o_file)

    ### 原系列と階差系列の比較表示 ###
    show_org_diff(ym_df['安値'], ym_diff['安値'])

    #print(df)
    #print(df.describe())


def show_trend_season(ser, o_file):

    def plot_aux(i, ind, val, title):

        plt.subplot(4,1,i)
        plt.plot(ind, val)
        plt.ylabel(title)
        plt.xlim(min_dt, max_dt)

    plt.figure(figsize=(8,5))
    ser.index = pd.to_datetime(ser.index)
    min_dt, max_dt = min(ser.index), max(ser.index)
    res = sm.tsa.seasonal_decompose(ser)
    plot_aux(1, ser.index, ser.values,  'ORIGINAL')
    plot_aux(2, ser.index, res.trend,   'TREND')
    plot_aux(3, ser.index, res.seasonal,'SEASONALITY')
    plot_aux(4, ser.index, res.resid,   'RESIDUAL')
    plt.subplots_adjust(hspace=1.0)
    plt.savefig(o_file)
    plt.close()


def show_acf(ser, o_file):

    #end_acf = sm.tsa.stattools.acf(ser, nlags=40)
    fig = plt.figure(figsize=(12,4))
    ax1 = fig.add_subplot(111)
    sm.graphics.tsa.plot_acf(ser, lags=40, ax=ax1)
    plt.savefig(o_file)
    plt.close()
    #plt.show()
    return


def show_org_diff(org_ser, diff_ser):

    plt.figure(figsize=(8,5))
    org_ser.index = pd.to_datetime(org_ser.index)
    min_dt, max_dt = min(org_ser.index), max(org_ser.index)
    plt.subplot(2,1,1)
    plt.plot(org_ser.index, org_ser.values)
    plt.ylabel('ORIGINAL')
    plt.subplot(2,1,2)
    plt.plot(diff_ser.index, diff_ser.values)
    plt.ylabel('DIFFERENCE')
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
