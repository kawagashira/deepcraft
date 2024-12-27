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
import pickle



def show_hist(ser, h_file):

    plt.hist(ser, bins=100)
    plt.savefig(h_file)
    plt.close()
    
    l_file = os.path.splitext(h_file)[0] + '-log.png'
    ser_log = np.log10(ser)
    plt.hist(ser_log, bins=100)
    plt.savefig(l_file)


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
    ax1 = fig.add_subplot(211)
    sm.graphics.tsa.plot_acf(ser, lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    sm.graphics.tsa.plot_pacf(ser, lags=40, ax=ax2)
    plt.subplots_adjust(hspace=0.5)

    plt.savefig(o_file)
    plt.close()
    return


def show_org_diff(org_ser, diff_ser, o_file):

    plt.figure(figsize=(8,5))
    org_ser.index = pd.to_datetime(org_ser.index)
    min_dt, max_dt = min(org_ser.index), max(org_ser.index)
    plt.subplot(2,1,1)
    plt.plot(org_ser.index, org_ser.values)
    plt.ylabel('ORIGINAL')
    plt.subplot(2,1,2)
    plt.plot(diff_ser.index, diff_ser.values)
    plt.ylabel('DIFFERENCE')
    plt.savefig(o_file)
    plt.close()


def adf_test(df):
    """
ADF検定
    """
    print('ADF Test: p-Value')
    res_ctt = sm.tsa.stattools.adfuller(df, regression="ctt")   # トレンド項あり（２次）、定数項あり
    res_ct  = sm.tsa.stattools.adfuller(df, regression="ct")   # トレンド項あり（１次、定数項あり
    res_c   = sm.tsa.stattools.adfuller(df, regression="c")   # トレンド項なし、定数項あり
    res_n   = sm.tsa.stattools.adfuller(df, regression="n")   # トレンド項なし、定数項なし
    
    print('ctt', '%0.4f' % res_ctt[1])
    print('ct ', '%0.4f' % res_ct[1])
    print('c  ', '%0.4f' % res_c[1])
    print('n  ', '%0.4f' % res_n[1])


def convert_to_float(df):

    ### 出来高 ###
    vol = np.array(list(map(to_volume, df['出来高'])),
        dtype=np.float32)
    df['出来高'] = vol

    ### 変化率 ###
    change = np.array(list(map(percent2value, df['変化率 %'])),
        dtype=np.float32)
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
    o_dir = 'fig'
    df = pd.read_csv(i_file,
        dtype={'日付け':'object',
            '終値':'float32', '始値':'float32', '高値':'float32', '安値':'float32',
            '出来高':'object', '変化率 %':'object'})

    ### 出来高、変化率をFloatに変換 ###
    df = convert_to_float(df)

    ### 欠損値の確認 ###
    print('欠損値')
    print(df.isnull().sum(axis=0))

    ### 頻度分布の確認 ###
    col = '安値'
    h_file = '%s/hist-%s.png' % (o_dir, col)
    show_hist(df[col], h_file)

    ### 月別データに変換 ###
    df['日付け'] = pd.to_datetime(df['日付け'])
    df['ym_dt'] = list(map(lambda dt: datetime.date(dt.year, dt.month, 1), df['日付け']))
    ym_df = df.groupby('ym_dt').mean()
    #ym_file = 'year_month.pkl'
    ym_file = 'year_month.csv'
    print('YEAR-MONTH DATA', ym_file)
    ym_df = ym_df.drop(columns='日付け')    # 月内の日付けの平均値は不要
    ym_df.to_csv(ym_file)

    ### トレンド、季節性の確認 ###
    if not os.path.isdir(o_dir):
        os.mkdir(o_dir)
    for col in ['始値', '終値', '安値', '高値']: 
        o_file = '%s/decompose-%s.png' % (o_dir, col)
        print('OUTPUT', o_file)
        show_trend_season(ym_df[col], o_file)

    col = '安値'
    ### 自己相関係数の確認 ###
    o_file = '%s/acf-%s.png' % (o_dir, col)
    print('ACF', o_file)
    show_acf(ym_df[col], o_file)

    ### 階差系列 ###
    ym_diff = (ym_df - ym_df.shift()).dropna()
    ym_diff_file = 'year_month_diff.csv'
    ym_diff.to_csv(ym_diff_file)
    o_file = '%s/diff-decomp-%s.png' % (o_dir, col)
    show_trend_season(ym_diff[col], o_file)

    ### 12ずらした階差系列 ##
    ym_diff12 = (ym_df - ym_df.shift(12)).dropna()
    o_file = '%s/diff12-asf-%s.png' % (o_dir, col)
    show_trend_season(ym_diff12[col], o_file)

    ### 原系列と階差系列の比較表示 ###
    o_file = '%s/comp-org-diff-%s.png' % (o_dir, col)
    show_org_diff(ym_df[col], ym_diff[col], o_file)

    ### 階差系列の自己相関係数 ###
    o_file = '%s/diff-acf-%s.png' % (o_dir, col)
    print('ACF', o_file)
    show_acf(ym_diff[col], o_file)

    ### ADF検定 ###
    col = '安値'
    adf_test(ym_df[col])

    ### パラメータ推定関数 ###
    res_selection = sm.tsa.arma_order_select_ic(ym_diff[col], ic='aic', trend='c')
    print(res_selection)
