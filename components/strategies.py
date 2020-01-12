#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 17:03:19 2020

@author: robin
"""

import numpy as np
import pandas as pd
from atomm.Indicators import MomentumIndicators

def MOM1(df, start_date, end_date, n_roc = 5, n_macd_short = 12, n_macd_long = 26, n_signal = 9, n_stoc = 7, n_rsi = 5):
    mi = MomentumIndicators(df)
    roc = mi.calcROC(n_roc)
 
    macd = mi.calcMACD(n_macd_short, n_macd_long)
    df2 = df.copy()
    df2['Close'] = macd
    signal = mi.calcEMA(n_signal, df2)
    df['SignalMACD'] = np.ones(len(df))
    df['SignalMACD'] = df['SignalMACD'].where(macd>signal).fillna(0)
    df['SignalROC'] = np.ones(len(df))
    df['SignalROC'] = df['SignalROC'].where(roc>0).fillna(0)
    ind_stoc = np.zeros(len(df))
    stoc = mi.calcSTOC(n_stoc)
    for i in range(len(stoc)):
        if (stoc[i] > 70) or (70 > stoc[i] > 30 and ind_stoc[i-1] == 1):
            ind_stoc[i] = 1
        else:   
            ind_stoc[i] = 0
    df['SignalSTOC'] = ind_stoc
    
    ind_rsi = np.zeros(len(df))
    rsi = mi.calcRSI(n_rsi)
    for i in range(len(rsi)):
        if (rsi[i] > 70) or (70 > rsi[i] > 30 and ind_rsi[i-1] == 1):
            ind_rsi[i] = 1
        else:
            ind_rsi[i] = 0
    df['SignalRSI'] = ind_rsi
    
    df['Signal'] = np.ones(len(df))
    sig = df['SignalMACD'] + df['SignalROC'] + df['SignalSTOC'] + df['SignalRSI']
    df['Signal'] = df['Signal'].where(sig>=3).fillna(0)
    df['bsSig'] = df['Signal'] - df['Signal'].shift(1) 
    
    returnsList = []
    for i in range(len(df)):
        sig1 = df['bsSig'].iat[i]
        val1 = df['Close'].iat[i]
        if sig1 == 1:
            priceBuy = val1
        elif sig1 == -1:
            priceSell = val1
            returns = (priceSell-priceBuy)/priceBuy
            returnsList.append(returns)

    returns_st = np.sum(returnsList)
    returns_bh = (df['Close'][-1]-df['Close'][0])/df['Close'][0]
    return df, returns_st, returns_bh
