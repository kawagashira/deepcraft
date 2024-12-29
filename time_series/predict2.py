#!/usr/bin/env python
#
#                           predict.py
#
import pickle
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

'''
def adf_test(df):
    """
ADF検定
    """
    print('ADF Test: p-Value')
    res_ctt = sm.tsa.stattools.adfuller(df, regression="ctt")   # トレンド項あり（２次）、定数項あり
    res_ct  = sm.tsa.stattools.adfuller(df, regression="ct")    # トレンド項あり(１次、定数項あり)
    res_c   = sm.tsa.stattools.adfuller(df, regression="c")     # トレンド項なし、定数項あり
    res_n   = sm.tsa.stattools.adfuller(df, regression="n")     # トレンド項なし、定数項なし

    print('ctt', '%0.4f' % res_ctt[1])
    print('ct ', '%0.4f' % res_ct[1])
    print('c  ', '%0.4f' % res_c[1])
    print('n  ', '%0.4f' % res_n[1])


def fit_sarima(ser, is_seasonal=False):

    if is_seasonal:    # With seasonal modelling
        seasonal_order=(1,1,1,12)
    else:           # Without seasonal modelling
        seasonal_order=(0,0,0,0)

    sarimax = sm.tsa.SARIMAX(ser,
        order=(2,2,1),
        #order=(2,1,1),
        #order=(1,1,1),
        seasonal_order=seasonal_order,
        enforce_stationarity    = False,
        enforce_invertibility   = False)
    model = sarimax.fit(maxiter=1000)
    print('AIC', model.aic)

    return model


def show_prediction(ser, pred, o_file):

    plt.figure(figsize=(8,4))
    plt.plot(ser.index, ser.values, label='original')
    plt.plot(pred.index, pred.values, label='predicted')
    plt.savefig(o_file)
    plt.close()


def evaluate(test, pred):

    wape = (test - pred).abs().sum() / test.sum()
    return wape


def show_org(ser, pred, o_file):

    last_id = pred.index[0]
    prev = ser[last_id]
    id_w, w = [], []
    for id, diff in pred[1:].items():
        integral = diff + prev 
        id_w.append(id)
        w.append(integral)
        prev = integral
    org_pred = pd.Series(w)
    org_pred.index = id_w

    ### SHOW ORGINAL DATA AND ITS PREDICTED VALUES ###
    plt.figure(figsize=(8,4))
    plt.plot(ser.index, ser.values, label='original')
    plt.plot(org_pred.index, org_pred.values, label='predicted')
    plt.savefig(o_file)
    plt.close()


def check_result(res, o_file):

    residuals = res.resid

    fig = plt.figure(figsize=(12,4))

    # 残差をプロット #
    ax1 = fig.add_subplot(211)
    ax1.plot(residuals)
    ax1.set_title('Residuals')
    ax2 = fig.add_subplot(212)
    sm.graphics.tsa.plot_pacf(residuals, lags=40, ax=ax2)
    ax2.set_title('Autocorrelation of Residuals')
    plt.subplots_adjust(hspace=0.5)

    # 残差の自己相関 #
    plt.savefig(o_file)
    plt.close()

'''

if __name__ == '__main__':

    from predict import adf_test, fit_sarima, show_prediction, evaluate, show_org, check_result
    o_dir = 'fig2'

    ### 原系列データ ###
    ym_file = 'year_month.csv'
    df = pd.read_csv(ym_file,
        index_col=0,
        dtype={'ym_dt':'object',
            '終値':'float32', '始値':'float32',
            '高値':'float32', '安値':'float32', '重心':'float32',
            '出来高':'float32', '変化率 %':'float32'})
    df.index = pd.to_datetime(df.index)

    ### 階差データ ###
    d_file = 'year_month_diff.csv'
    df_diff = pd.read_csv(d_file,
        index_col=0,
        dtype={'ym_dt':'object',
            '終値':'float32', '始値':'float32',
            '高値':'float32', '安値':'float32', '重心':'float32',
            '出来高':'float32', '変化率 %':'float32'})
    df_diff.index = pd.to_datetime(df_diff.index)

    print(df_diff.dtypes)
    print(df_diff)
    col = '重心'
    ser = df_diff[col]     # 階差系列データを使用

    ### ADF検定 ###
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    print('階差系列でのADF検定')
    #ser = ser/100
    adf_test(ser)

    ### パラメータ推定関数 ###
    res_selection = sm.tsa.arma_order_select_ic(ser,
        ic='aic', trend='c', max_ar=3, max_ma=3)
    print(res_selection.aic_min_order)

    ### 訓練と評価データに分割 ###
    n = 56
    train, test = ser[:-n], ser[-n:]
    print('train', len(train), 'test', len(test))

    ### 季節性なしSARIMA ###
    model = fit_sarima(train, is_seasonal=False)
    pred = model.predict('2019-12-01', '2024-08-01')
    print('pred.index', type(pred.index))
    p_file = '%s/show_prediction-%s-noseasonal.png' % (o_dir, col)
    print('季節性なし予測結果表示', p_file)
    r_file = '%s/residual-acf-%s-noseasonal.png' % (o_dir, col)
    print('季節性なしACF', r_file)
    show_prediction(ser, pred, p_file)
    print('季節性なしWAPE', evaluate(test, pred[1:]))

    ### 季節性ありSARIMA ###
    model = fit_sarima(train, is_seasonal=True)
    pred = model.predict('2019-12-01', '2024-08-01')
    p_file = '%s/show_prediction-%s-seasonal.png' % (o_dir, col)
    print('季節性あり予測結果表示', p_file)
    r_file = '%s/residual-acf-%s-seasonal.png' % (o_dir, col)
    print('季節性ありACF', r_file)
    show_prediction(ser, pred, p_file)

    ### 評価 ###
    #pred = pred[1:]     # 最初のオーバーラップ点を除去する
    print('季節性ありWAPE', evaluate(test, pred[1:]))

    ### 原系列と予測を表示
    g_file = '%s/show_orig_pred-%s' % (o_dir, col)
    show_org(df[col], pred, g_file)

    check_result(model, r_file)

