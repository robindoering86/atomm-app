#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 16:48:42 2020

@author: robin
"""

import pandas as pd
import numpy as np
import plotly.graph_objs as go

from atomm.Indicators import MomentumIndicators
from atomm.DataManager.main import MSDataManager 
import colorlover as cl
colorscale = cl.scales['12']['qual']['Paired']
cgreen = 'rgb(12, 205, 24)'
# cgreen = 'rgb(16, 30, 30)'
# cgreen = 'rgb(45, 76, 80)'

cred = 'rgb(205, 12, 24)'
# cred = 'rgb(255, 114, 33)'


####
# Plot prices, daily returns etc.
####
def d_close(df):
    trace = go.Scatter(
        x = df.index,
        y = df['Close'],
        mode = 'lines',
        name = 'Daily Close',
        showlegend = False,
        line = {'width': 2}
        )
    return trace

def ohlc(df):
    trace = go.Candlestick(x = df.index,
                           open = df['Open'],
                           high = df['High'],
                           low=df['Low'],
                           close=df['Close'],
                           showlegend = False,
                           increasing_line_color= cgreen,
                           decreasing_line_color= cred
                           )
    return trace

def average_close(df, index, ticker, fig):
    #dh = am.DataHandler(index)
    dh = MSDataManager()
    data = None
    #data = dh.ReturnDataRT(ticker)
    #if not data.empty
    if data is not None:
        ftwh = data['52_week_high'].iloc[-1]
        ftwl = data['52_week_low'].iloc[-1]
        trace = go.Scatter(x = df.index, y = (np.zeros(len(df.index)) + ftwh), mode = 'lines', name = '52wh', showlegend = False, line = {'color': 'rgb(255, 255, 255)', 'width': 1, 'dash': 'dot'})
        fig.append_trace(trace, 1, 1)
        trace = go.Scatter(x = df.index, y = (np.zeros(len(df.index)) + df['Close'].mean()), mode = 'lines', name = 'Average', showlegend = False, line = {'color': 'rgb(255, 255, 255)', 'width': 1, 'dash': 'dot'})
        fig.append_trace(trace, 1, 1)
        trace = go.Scatter(x = df.index, y = (np.zeros(len(df.index)) + ftwl), mode = 'lines', name = '52wl', showlegend = False, line = {'color': 'rgb(255, 255, 255)', 'width': 1, 'dash': 'dot'})
        fig.append_trace(trace, 1, 1)
    return fig

def bsMarkers(df, fig):
#    df['Long'] = np.zeros(len(df))
#    df['Short'] = np.zeros(len(df))
#    
#    signalCol = df.columns.get_loc('Signal')
#    closeCol = df.columns.get_loc('Close')
#    longCol = df.columns.get_loc('Long')
#    shortCol = df.columns.get_loc('Short')
#    for i in range(len(df)):
#        sig1 = df['Signal'].iat[i]
#        sig2 = df['Signal'].iat[i-1]
#        val1 = df['Close'].iat[i]
#        if sig1 == 1:
#            df['Long'].iat[i] = val1
#            df['Short'].iat[i] = 0
#        elif sig1 == 0 and sig2 == 1:
#            df['Long'].iat[i] = val1
#            df['Short'].iat[i] = val1
#        elif sig1 == 1 and sig2 == 0:
#            df['Long'].iat[i] = 0
#            df['Short'].iat[i] = val1    
#        else:
#            df['Long'].iat[i] = 0
#            df['Short'].iat[i] = val1
#    long = go.Scatter(x = df.index, y = df['Long'], connectgaps = False, name = 'long', mode='lines+markers',fill='tozeroy', line = {'color': cgreen, 'width': 2})
#    short = go.Scatter(x = df.index, y = df['Short'], connectgaps = False, name = 'long', mode='lines+markers', fill='tozeroy', line = {'color': cred, 'width': 2})
            
            
    long = go.Scatter(x = df.index, y = df['Close'].where(df['Signal'] == 1, None), showlegend = False, connectgaps = False, name = 'long', mode='lines', line = {'color': cgreen, 'width': 2})
    fig.append_trace(long, 1, 1)
    short = go.Scatter(x = df.index, y = df['Close'].where(df['Signal'] == 0, None), showlegend = False, connectgaps = False, name = 'long', mode='lines', line = {'color': cred, 'width': 2})
    fig.append_trace(short, 1, 1)
    buy = go.Scatter(x = df.loc[df['bsSig'] == 1 , 'Close'].index, y = df.loc[df['bsSig'] == 1, 'Close'].values, showlegend = False, name = 'buy', mode='markers', marker = {'size': 15, 'color': cgreen, 'opacity': 0.75, 'symbol': 'triangle-up'})
    fig.append_trace(buy, 1, 1)
    sell = go.Scatter(x = df.loc[df['bsSig'] == -1 , 'Close'].index, y = df.loc[df['bsSig'] == -1, 'Close'].values, showlegend = False, name = 'sell', mode='markers', marker = {'size': 15, 'color': cred, 'opacity': 0.75, 'symbol': 'triangle-down'})
    fig.append_trace(sell, 1, 1)
    return fig

def vol_traded(df, fig, row):
    trace = go.Scatter(
        x = df.index,
        y = df['Volume'],
        showlegend = False,
        line = {'color': 'rgb(219, 240, 238)', 'width': 1},
        fillcolor = 'rgba(219, 240, 238, 0.5)', fill = 'tozeroy', name = 'Vol. Trad.')
    fig.append_trace(trace, row, 1)
    return fig

def green_ref_line(df, y_pos, name):
    return (go.Scatter(x = df.index, y = (np.zeros(len(df.index)) + y_pos), name = name, showlegend = False, line = {'color': cgreen, 'width': 1}, hoverinfo='skip'))

def red_ref_line(df, y_pos, name):
    return (go.Scatter(x = df.index, y = (np.zeros(len(df.index)) + y_pos), name = name, showlegend = False, line = {'color': cred, 'width': 1}, hoverinfo='skip'))
####
# Momentum Strategies
####

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

####
# Calculate Studies
####


####
# Studies
####
def EMA10(df, fig):
    mi = MomentumIndicators(df)
    trace = (go.Scatter(x = df.index, y = mi.EMA(10), showlegend = False, name = 'EMA10'))
    fig.append_trace(trace, 1, 1)
    return fig    
def EMA30(df, fig):
    mi = MomentumIndicators(df)
    trace = (go.Scatter(x = df.index, y = mi.calcEMA(30), showlegend = False, name = 'EMA10'))
    fig.append_trace(trace, 1, 1)
    return fig

def SMA10(df, fig):
    mi = MomentumIndicators(df)
    trace = (go.Scatter(x = df.index, y = mi.calcSMA(10), showlegend = False, name = 'SMA10'))
    fig.append_trace(trace, 1, 1)
    return fig

def SMA30(df, fig):
    mi = MomentumIndicators(df)
    trace = (go.Scatter(x = df.index, showlegend = False, y = mi.calcSMA(30), name = 'SMA30'))
    fig.append_trace(trace, 1, 1)
    return fig

def BB202(df, fig):
    mi = MomentumIndicators(df)
    # sma20 = mi.calcSMA(20)
    # std = mi.calcSTD(20)
    upper, lower, sma20 = mi.calcBB(20, 2)
    trace = (go.Scatter(
        x = df.index,
        y = sma20,
        name = 'BB(20, 2)',
        line = {
            #'color': 'rgb(250, 250, 250)',                
            'width': 1
            },
        showlegend = False)
        )
    fig.append_trace(trace, 1, 1)
    trace = (go.Scatter(
        x = df.index,
        y = lower,
        name = 'BB(20, 2)',
        line = {
            'color': 'rgb(219, 240, 238)',                
            'width': 1
            },
        showlegend = False
        ))
    fig.append_trace(trace, 1, 1)
    trace = (go.Scatter(
        x = df.index,
        y = upper,
        name = 'BB(20, 2)',
        showlegend = False,
        line = {
            'color': 'rgb(219, 240, 238)',                
            'width': 1
            },
        fill='tonexty',
        fillcolor = 'rgba(219, 240, 238 0.5)'
        ))
    fig.append_trace(trace, 1, 1)
    return fig

def D_RETURNS(df, fig, row):
    d_return = df['Close'].pct_change()
    fig.append_trace((go.Scatter(
        x = df.index,                                
        y = d_return.where(d_return >= 0).fillna(0),
        name = 'Daily return', mode = 'lines',
        connectgaps = True,
        showlegend = False,
        line = {'width': 0}, 
        fill='tozeroy',
        fillcolor = 'rgba(12, 205, 24, 0.3)')
        ), row, 1)
    fig.append_trace((go.Scatter(
        x = df.index,
        y = d_return.where(d_return < 0).fillna(0),
        name = 'Daily return',
        mode = 'lines',
        connectgaps = True,
        showlegend = False,
        line = {'width': 0},
        fill='tozeroy',
        fillcolor = 'rgba(205, 12, 24, 0.3)')
        ), row, 1)
    fig.append_trace((go.Scatter(
        x = df.index,
        y = d_return,
        name = 'Daily return',
        mode = 'lines', 
        line = {'color': 'rgb(250, 250, 250)', 'width': 1},
        showlegend = False,)
        ), row, 1)
    fig.append_trace((go.Scatter(
        x = df.index,
        y = (np.zeros(len(df.index)) + d_return.mean()),
        name = 'Avg. daily return', 
        showlegend = False,
        mode = 'lines', 
        line = {'color': 'rgb(250, 250, 250)', 'width': 1})
        ), row, 1)
    fig['layout']['yaxis' + str(row)].update(autorange = True,
       showgrid = True,
       title  = 'Daily returns',
       gridcolor = 'rgba(255, 255, 255, 0.3)',
       color = 'rgba(255, 255, 255, 1)',
       )    
    return fig

def RSI(df, fig, row):
    mi = MomentumIndicators(df)
    lower = 30
    upper = 70
    n = 5
    RSI1 = mi.calcRSI(n)
    fig.append_trace(green_ref_line(df, upper, 'RSIREF80'), row, 1)
    fig.append_trace((go.Scatter(x = df.index, y = RSI1.where(RSI1 >= upper).fillna(upper), name = 'RSI', mode = 'lines', showlegend = False, line = {'width': 0}, fill='tonexty', fillcolor = 'rgba(12, 205, 24, 0.3)', hoverinfo='skip')), row, 1)
    fig.append_trace(red_ref_line(df, lower, 'RSIREF80'), row, 1) 
    fig.append_trace((go.Scatter(x = df.index, y = RSI1.where(RSI1 < lower).fillna(lower), name = 'RSI', mode = 'lines', showlegend = False, line = {'width': 0}, fill='tonexty', fillcolor = 'rgba(205, 12, 24, 0.3)', hoverinfo='skip')), row, 1)
    fig.append_trace((go.Scatter(x = df.index, y = RSI1, name = 'RSI', mode = 'lines', showlegend = False, line = {'color': 'rgb(250, 250, 250)', 'width': 1})), row, 1)
    fig['layout']['yaxis' + str(row)].update(autorange = True,
       showgrid = True,
       title  = 'RSI(' + str(n) + ') (%)',
       gridcolor = 'rgba(255, 255, 255, 0.3)',
       color = 'rgba(255, 255, 255, 1)',
       tickmode = 'array',
       tickvals = [30, 70],
       zeroline = False
       )    
    return fig

def ROC(df, fig, row):
    mi = MomentumIndicators(df)
    n = 5
    ROC = mi.calcROC(n)   
    fig.append_trace((go.Scatter(x = df.index, y = ROC.where(ROC >= 0).fillna(0), name = 'ROC(5)', mode = 'lines', connectgaps = True, showlegend = False, line = {'width': 0}, fill='tozeroy', fillcolor = 'rgba(12, 205, 24, 0.3)')), row, 1)
    fig.append_trace((go.Scatter(x = df.index, y = ROC.where(ROC < 0).fillna(0), name = 'ROC(5)', mode = 'lines', connectgaps = True, showlegend = False, line = {'width': 0}, fill='tozeroy', fillcolor = 'rgba(205, 12, 24, 0.3)')), row, 1)
    fig.append_trace((go.Scatter(x = df.index, y = ROC, name = 'ROC(5)', mode = 'lines', showlegend = False, line = {'color': 'rgb(250, 250, 250)', 'width': 1})), row, 1)
    fig['layout']['yaxis' + str(row)].update(autorange = True,
       showgrid = True,
       title  = 'ROC(' + str(n) + ')',
       gridcolor = 'rgba(255, 255, 255, 0.3)',
       color = 'rgba(255, 255, 255, 1)',
       )    
    return fig


def MACD(df, fig, row):
    mi = MomentumIndicators(df)
    n_fast = 12
    n_slow = 26
    macd = mi.calcMACD(n_fast, n_slow)
    df2 = df.copy()
    df2['Close'] = macd
    signal = mi.calcEMA(9, df2)
    fig.append_trace((go.Scatter(x = df.index, y = macd.where(macd >= 0).fillna(0), name = 'MACD(12,26)', mode = 'lines', connectgaps = True, showlegend = False, line = {'width': 0}, fill='tozeroy', fillcolor = 'rgba(12, 205, 24, 0.3)')), row, 1)
    fig.append_trace((go.Scatter(x = df.index, y = macd.where(macd < 0).fillna(0), name = 'MACD(12,26)', mode = 'lines', connectgaps = True, showlegend = False, line = {'width': 0}, fill='tozeroy', fillcolor = 'rgba(205, 12, 24, 0.3)')), row, 1)
    fig.append_trace((go.Scatter(x = df.index, y = macd, name = 'MACD(12,26)', mode = 'lines', showlegend = False, line = {'color': 'rgb(250, 250, 250)', 'width': 1})), row, 1)
    fig.append_trace((go.Scatter(x = df.index, y = signal, name = 'EMA(9)', mode = 'lines', showlegend = False, line = {'color': 'rgb(120, 120, 250)', 'width': 1})), row, 1)
    fig['layout']['yaxis' + str(row)].update(autorange = True,
       showgrid = True,
       title  = 'MACD(' + str(n_fast) + ', ' + str(n_slow) + ')',
       gridcolor = 'rgba(255, 255, 255, 0.3)',
       color = 'rgba(255, 255, 255, 1)',
       )    
    return fig

def STOC(df, fig, row):
    mi = MomentumIndicators(df)
    lower = 30
    upper = 70
    n = 7
    stoc = mi.calcSTOC(n)
    fig.append_trace(green_ref_line(df, upper, 'STOCREF' + str(upper)), row, 1)
    fig.append_trace((go.Scatter(x = df.index, y = stoc.where(stoc >= upper).fillna(upper), name = 'STOC(' + str(n)  + ')', mode = 'lines', showlegend = False, line = {'width': 0}, fill='tonexty', fillcolor = 'rgba(12, 205, 24, 0.3)')), row, 1)
    fig.append_trace(red_ref_line(df, lower, 'STOCREF' + str(lower)), row, 1) 
    fig.append_trace((go.Scatter(x = df.index, y = stoc.where(stoc < lower).fillna(lower), name = 'STOC(' + str(n) + ')', mode = 'lines', showlegend = False, line = {'width': 0}, fill='tonexty', fillcolor = 'rgba(205, 12, 24, 25)')), row, 1)
    fig.append_trace((go.Scatter(x = df.index, y = stoc, name = 'STOC(' + str(n) + ')', mode = 'lines', showlegend = False, line = {'color': 'rgb(250, 250, 250)', 'width': 1})), row, 1)
    return fig

def ATR(df, fig, row):
    mi = MomentumIndicators(df)
    n = 5
    atr = mi.calcATR(n)
    fig.append_trace(go.Scatter(x = df.index, y = atr, name = 'ATR(' + str(n) + ')', mode = 'lines', showlegend = False, line = {'color': 'rgb(250, 250, 250)', 'width': 1}), row, 1)
    fig['layout']['yaxis' + str(row)].update(autorange = True,
       showgrid = True,
       title  = 'ATR(' + str(n) + ')',
       gridcolor = 'rgba(255, 255, 255, 1)',
       color = 'rgba(255, 255, 255, 1)',
       tickmode = 'array',
       tickvals = [30, 70],
       zeroline = True
       )    
    return fig

def ADX(df, fig, row):
    mi = MomentumIndicators(df)
    n = 10
    adx = mi.calcADX(n)
    fig.append_trace(go.Scatter(
        x = df.index, 
        y = adx, 
        name = (f'ADX({n})'), 
        mode = 'lines', 
        showlegend = False, 
        line = {
            'color': 'rgb(250, 250, 250)',
            'width': 1})
        , row, 1)
    fig['layout']['yaxis' + str(row)].update(autorange = True,
       showgrid = True,
       title  = (f'ADX({n})'),
       gridcolor = 'rgba(255, 255, 255, 1)',
       color = 'rgba(255, 255, 255, 1)',
       tickmode = 'array',
       tickvals = [30, 70],
       zeroline = True
       )    
    return fig

def WR(df, fig, row):
    mi = MomentumIndicators(df)
    n = 10
    wr = mi.calcWR(n)(n)
    fig.append_trace(go.Scatter(
        x = df.index, 
        y = wr, 
        name = (f'WR({n})'), 
        mode = 'lines', 
        showlegend = False, 
        line = {
            'color': 'rgb(250, 250, 250)',
            'width': 1})
        , row, 1)
    fig['layout']['yaxis' + str(row)].update(autorange = True,
       showgrid = True,
       title  = (f'WR({n})'),
       gridcolor = 'rgba(255, 255, 255, 1)',
       color = 'rgba(255, 255, 255, 1)',
       tickmode = 'array',
       tickvals = [30, 70],
       zeroline = True
       )    
    return fig

    
    

# [,'williamsr', 'cci', , 'log_ret', 'adx' ,
#             'stocd', ' 'autocorr_1',\
#            'autocorr_3','autocorr_5']

#####
#
#####