#!/usr/bin/env python
#
#                           predict2.py
#

import pandas as pd
import statsmodels.api as sm

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

