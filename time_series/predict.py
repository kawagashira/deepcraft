#!/usr/bin/env python
#
#                           predict.py
#
import pickle
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

def predict_sarima(ser):

    print(ser)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    ser_scaled = scaler.fit_transform(ser.values.reshape(1,-1)).flatten()

    sarimax = sm.tsa.SARIMAX(ser,
    #sarimax = sm.tsa.SARIMAX(ser_scaled,
        #order=(6,2,1),
        #order=(5,2,1),
        order=(4,2,1),
        #order=(3,2,1),
        #order=(2,2,1),
        #seasonal_order=(1,1,1,12),
        seasonal_order=(0,0,0,0),
        enforce_stationarity    = False,
        enforce_invertibility   = False,
        )
    result = sarimax.fit(maxiter=1000)
    #print('モデルの残差成分')
    #print(result.resid)
    print('AIC', result.aic)
    #print(result.mle_retvals)

    return result


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
    #from statsmodels.graphics.tsaplots import plot_acf
    plt.savefig(o_file)
    plt.close()


if __name__ == '__main__':

    o_dir = 'fig'
    ### 原系列データ ###
    i_file = 'year_month.pkl'
    with open(i_file, 'rb') as i_handle:
        df = pickle.load(i_handle)
    print(df)

    ### 階差データ ###
    d_file = 'year_month_diff.pkl'
    with open(d_file, 'rb') as d_handle:
        df_diff = pickle.load(d_handle)

    ### ADF検定 ###

    col = '安値'
    res = predict_sarima(df[col])
    #res = predict_sarima(np.log10(df[col]))
    #res = predict_sarima(df_diff[col])

    r_file = '%s/residual-acf-%s' % (o_dir, col)
    print('ACF', r_file)
    check_result(res, r_file)

