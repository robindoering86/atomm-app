#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Created on Tue Jul  2 09:29:34 2019

@author: robin
'''

import dash
#import dash_auth
import dash_core_components as dcc
import dash_html_components as html
import dash_table as ddt
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objs as go
from plotly import tools
#import plotly.io as pio
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime
#from flask_cachi1ng import Cache
#import quandl as quandl 
import atomm as am
import atomm.Tools as to
from atomm.Indicators import MomentumIndicators
from atomm.DataManager.main import MSDataManager
from atomm.Tools import MarketHours
#to = am.Tools()
#import time

app = dash.Dash()
theme = 'seaborn'

#cache = Cache(app.server, config={
#    'CACHE_TYPE': 'filesystem',
#    'CACHE_DIR': 'cache-directory'
#})

import colorlover as cl
colorscale = cl.scales['12']['qual']['Paired']

cgreen = 'rgb(12, 205, 24)'
cred = 'rgb(205, 12, 24)'


# Set number of allowed charts
num_charts = 4

server = app.server

#conf = am.Config()


#dh = am.DataHandler('DAX')
dh = MSDataManager()
options = dh.IndexConstituentsDict('SPY')
indicators = [
                    {'label': '10 day exponetial moving average', 'value': 'EMA10'},
                    {'label': '30 day exponetial moving average', 'value': 'EMA30'},
                    {'label': '10 day Simple moving average', 'value': 'SMA10'},
                  {'label': '30 day Simple moving average', 'value': 'SMA30'},
                  {'label': 'Bollinger Bands (20, 2)', 'value': 'BB202'},
                  {'label': 'RSI', 'value': 'RSI'},
                  {'label': 'ROC', 'value': 'ROC'},
                  {'label': 'MACD(12, 26)', 'value': 'MACD'},
                  {'label': 'STOC(7)', 'value': 'STOC'},
                  {'label': 'ATR(7)', 'value': 'ATR'},
                  {'label': 'Daily returns', 'value': 'D_RETURNS'}
              ]

strategies = [{'label': 'Momentum follower 1', 'value': 'MOM1'}]


def create_chart_div(num):
    divs = (    html.Div(
                id = str(num) + 'graph_div',
                className = 'visible chart-div one-column',
                children = [
                    html.Div([
                            html.Div(
                                    children = [html.H1('Hello World')],   
                                    id = str(num) + 'info_div',
                                    className = 'menu_bar_left'),
                            html.Div(
                                    html.Button('', id = 'close_button' + str(num) + 'graph_div', n_clicks = 0, n_clicks_timestamp = 0),
                                    className = 'menu_bar_right', style = {'display': 'none'}),
                        ],
                        className = 'menu_bar'
                        ),
                    html.Div([html.H3('Stock symbol', style = {'display': 'inline-block'}),
                              dcc.Dropdown(id = str(num) + 'stock_picker',
                                           options = options,
                                           multi = False,
                                           value = 'AAPL',
                                           placeholder = 'Enter stock symbol',
                                           className = 'dash-bootstrap'
                        )],
                        style = {'display': 'inline-block', 'verticalAlign': 'top'},
                        id = str(num) + 'stock_picker_div',
                        className = 'stock_picker'    
                    ),
                    html.Div([html.H3('Studies', style = {'display': 'none'}),
                            dcc.Dropdown(id = str(num) + 'study_picker',
                                      options = indicators,
                                      multi = True,
                                      value = [],
                                      placeholder = 'Select studies',
                                      className = 'dash-bootstrap'
                        )], 
                        style = {'display': 'inline-block', 'verticalAlign': 'top'},
                        id = str(num) + 'study_picker_div',
                        className = 'study_picker'
                    ),
                    html.Div([html.H3('Strategy', style = {'display': 'none'}),
                            dcc.Dropdown(id = str(num) + 'strategyTest',
                                      options = strategies,
                                      multi = False,
                                      value = '',
                                      placeholder = 'Test strategies',
                                      className = 'dash-bootstrap'
                        )], 
                        style = {'display': 'inline-block', 'verticalAlign': 'top'},
                        id = str(num) + 'strategie_picker_div',
                        className = 'strategie_picker'
                    ),

#                    html.Div([dcc.RadioItems(id = str(num) + 'timeRangeSel',
#                                             options=[
#                                                     {'label': '1 W', 'value': '1w'},
#                                                     {'label': '1 M', 'value': '1m'},
#                                                     {'label': '3 M', 'value': '3m'},
#                                                     {'label': '6 M', 'value': '6m'},
#                                                     {'label': '1 Y', 'value': '12m'},
#                                                     {'label': '3 Y', 'value': '36m'},
#                                                     {'label': 'Max', 'value': 'max'}
#                                                     ],
#                                             value = '6m',
#                                             labelStyle = {'display': 'inline-block'},
#                                             className = 'switch-field',
#                                             labelClassName = 'time-range-label',
#                                             inputClassName = 'time-range-input'
#                                             )
#                              ],
#                    style = {'display': 'inline-block'},
#                    id = str(num) + 'timeRangeSel_div'
#                    ),
                    html.Div(
                        dcc.RadioItems(
                            id = str(num) + 'trace_style',
                            options=[
                                {'label': 'Line Chart', 'value': 'd_close'},
                                {'label': 'Candlestick', 'value': 'ohlc'}
                            ],
                            value='ohlc',
                        ),
                        id=str(num) + 'style_tab',
                        style = {'marginTop': '0', 'textAlign': 'left', 'display': 'inline-block'}
                    ),
                    html.Div(
                            dcc.Graph(
                                    id = str(num) + 'chart',
                                    figure = {
                                            'data': [],
                                            'layout': [{'height': 900}]
                                            },
                                    config = {'displayModeBar': False, 'scrollZoom': True, 'fillFrame': False},
                            ),
                            id = str(num) + 'graph',
                            )
                            ]
                       )
            )
    return divs

def create_sidebar_tab(name):
    tab = dcc.Tab(label=name, children=[
        html.Div([]          
                )                      
          ],
          id = 'tab-' + str(name),
          value='tab-' + str(name),
          className='custom-tab',
          selected_className='custom-tab--selected'  
          )
    return tab


def create_sidebar_tabs():
    tabs = html.Div([
                dcc.Tabs(
                        id = 'stockIndexTabs',
                        value = 'tab-SPY',
                        parent_className = 'custom-tabs',
                        className = 'custom-tabs-container',
                        children = [
                                create_sidebar_tab(index) for index in dh.IndexList()]
                        ),
                html.Div(
                                [
                                        html.Div([        
                                                dcc.Loading(id="loading-1", children=[
                                                    ddt.DataTable(
                                                        id = 'IndexListTable',
                                                        data = [] ,                   
                                                        columns = [{'id': 'Symbol', 'name': 'Symbol'}, {'id': 'Name', 'name': 'Name'}, {'id': '52ws', 'name': '52w Sharpe (%)'}, {'id': '52wh', 'name': '52w high'}, {'id': '52wl', 'name': '52w low'} ],
                                                        filter_action = "native",
                                                        sort_action = "native",
                                                        sort_mode = "multi",
                                                        row_selectable = False,
                                                        style_table = {
                                                                'maxHeight': '900px',
                                                                'overflowY': 'scroll'
                                                                },
                                                                
                                                        style_cell={
                                                                'padding': '3px 2px',
                                                                'width': 'auto',
                                                                'textAlign': 'center'
                                                                },
                                                        style_cell_conditional=[
                                                                {
                                                                    'if': {'row_index': 'even'},
                                                                    'backgroundColor': '#f9f9f9'
                                                                }
                                                            ],
                                                                
                                                    )
                                                ],
                                                type="default"
                                                )
                                                ],
                                                id = 'table_container_div',
                                                ),                      
#                                        
                                        html.Div([
                                        html.Div([
                                                html.Button('Full Market', id = 'full_market_backtest', n_clicks = 0)
                                                ],
                                                id='slider-output-container',
                                                style = {'display': 'block', 'width': '20%'},
                                                className = ''
                                                ),
                                        dcc.Loading(
                                                id = 'loading-2',
                                                children=[
                                                        html.Div(
                                                            children = [],
                                                            id='full-market-backtest-results',
                                                            style = {'display': 'block', 'width': '80%'},
                                                            className = ''
                                                        )
                                                ]
                                        )
                                         
                                             ]
                                            )
                                        ]
                                        )
                        
                    ])
    return tabs


def create_strategyTestResult(num):
    data = html.Div(
            className = 'visible chart-div two-column',
            id = str(num) + 'strategyTestResults_container_div',
            children = [
                    html.H2('Strategy Results'),
                    html.Div([dcc.Graph(id = 'hist-graph', clear_on_unhover = True)]),
                    ddt.DataTable(
                            id = str(num) + 'strategyTestResults',
                            data = [],
                            columns = [
                                {'id': 'Symbol', 'name': 'Symbol', 'type': 'text'}, 
                                {'id': 'datebuy', 'name': 'Buy Date', 'type': 'datetime'},
                                {'id': 'pricebuy', 'name': 'Buy Price', 'type': 'numeric'},
                                {'id': 'buysignals', 'name': 'Buy Sig. (M,ROC,STOC,RSI)', 'type': 'text'}, 
                                {'id': 'datesell', 'name': 'Sell Date', 'type': 'datetime'},
                                {'id': 'pricesell', 'name': 'Sell Price', 'type': 'numeric'},
                                {'id': 'sellsignals', 'name': 'Sell Sig. (M,ROC,ST,RSI)', 'type': 'text'},
                                {'id': 'perprofit', 'name': '% Profit', 'type': 'numeric'}],
                            filter_action = 'native',
                            sort_action = 'native',
                            sort_mode = 'multi',
                            row_selectable = False,
                            style_table={'overflowX': 'scroll'}
#                            style_data={'whiteSpace': 'normal'},
#                            css=[{
#                                'selector': '.dash-cell div.dash-cell-value',
#                                'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
#                            }]
                        )
                        ]
            )                      
    
    return data


def create_stockList(num):
    df = pd.read_csv('./Data/WTD-stocklist.csv')
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    stockData = df.to_dict('records')
#    print(df[df['stock_exchange_short'] == 'GER' and df['currency'] == 'EUR'])
    data = html.Div(
            className = 'visible chart-div two-column',
            id = 'stockList_container_div',
            children = [
                    html.H2('Stock List'),
                    ddt.DataTable(
                            id = 'stockList',
                            data = stockData,
                            columns = [{'id': nam, 'name': nam} for nam in df.columns],
                            filter_action = 'native',
                            sort_action = 'native',
                            sort_mode = 'multi',
                            row_selectable = False,
#                            style_table={'overflowX': 'scroll'}
                            style_data={'whiteSpace': 'normal'},
                            css=[{
                                'selector': '.dash-cell div.dash-cell-value',
                                'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
                            }]
                        )
                        ]
            )                      
    
    return data


########
##
## MAIN DASHBOARD LAYOUT
##
########
app.layout = html.Div([
        html.Div([
                html.Div(
                        [
                                html.H1('atomm trading app'),
                        ],
                        className='content_header'
                        ),
                dcc.Interval(
                        id='interval-component',
                        interval=30*1000,
                        n_intervals=0
                        ),
                html.Div(
                        [
                                create_chart_div(i) for i in range(0, 1)
                        ],
                        id='charts'
                        ),
                html.Div(
                        [
                                create_strategyTestResult(0)
                        ]
                        ),
                #html.Div(
                #        [
                #                create_stockList(0)
                #        ]
                #        )
                ],
                id='content'
                ),
        html.Div(
                [
                        html.Div(
                                [
                                        html.H1('Markets')
                                        ],
                                id='sidebar_right_header',
                                className='sidebar_right_header'
                        ),
                        html.Div(
                                [
                                        create_sidebar_tabs()
                                        ]
                        ),
                ],
                id='sidebar_right',
                className='right_menu'
                ),
        html.Div(
            id='charts_clicked',
            style={'display': 'none'}
            ),
        html.Div(
            id='selected-index-div',
            style={'display': 'inline-block'}
            ),
        html.Div(
            id='div-out',
            style={'display': 'inline-block'}
            ),
        html.Div(
            id='add_btn_counter',
            style={'display': 'none'}
            ),
        ])



####
# Plot prices, daily returns etc.
####
stoc.zeros(len(df.index)) + y_pos), name = name, showlegend = False, line = {'color': cred, 'width': 1}, hoverinfo='skip'))
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
    sma20 = mi.calcSMA(20)
    std = mi.calcSTD(20)
    trace = (go.Scatter(x = df.index, y = sma20, name = 'BB(20, 2)', showlegend = False))
    fig.append_trace(trace, 1, 1)
    trace = (go.Scatter(x = df.index, y = sma20-2*std, name = 'BB(20, 2)', showlegend = False))
    fig.append_trace(trace, 1, 1)
    trace = (go.Scatter(x = df.index, y = sma20+2*std, name = 'BB(20, 2)', showlegend = False))
    fig.append_trace(trace, 1, 1)
    return fig

def D_RETURNS(df, fig, row):
    d_return = df['Close'].pct_change()
    fig.append_trace((go.Scatter(x = df.index, y = d_return.where(d_return >= 0).fillna(0), name = 'Daily return', mode = 'lines', connectgaps = True, showlegend = False, line = {'width': 0}, fill='tozeroy', fillcolor = 'rgba(12, 205, 24, 0.3)')), row, 1)
    fig.append_trace((go.Scatter(x = df.index, y = d_return.where(d_return < 0).fillna(0), name = 'Daily return', mode = 'lines', connectgaps = True, showlegend = False, line = {'width': 0}, fill='tozeroy', fillcolor = 'rgba(205, 12, 24, 0.3)')), row, 1)
    fig.append_trace((go.Scatter(x = df.index, y = d_return, name = 'Daily return', mode = 'lines', line = {'color': 'rgb(250, 250, 250)', 'width': 1}, showlegend = False,)), row, 1)
    fig.append_trace((go.Scatter(x = df.index, y = (np.zeros(len(df.index)) + d_return.mean()), name = 'Avg. daily return', showlegend = False, mode = 'lines', line = {'color': 'rgb(250, 250, 250)', 'width': 1})), row, 1)
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

#####
#
#####
def get_fig(ticker, type_trace, studies, strategyTest, start_date, end_date, index):

#    for ticker in ticker_list:
    #dh = am.DataHandler(index)
    dh = MSDataManager()
    df = dh.ReturnData(ticker, start_date=start_date, end_date=end_date)

    subplot_traces = [  # first row traces
        'RSI',
        'ROC',
        'MACD',
        'STOC',
        'D_RETURNS',
        'ATR'
    ]
    selected_subplots_studies = []
    selected_first_row_studies = []
    row = 2  # number of subplots

    if studies != []:
        for study in studies:
            if study in subplot_traces:
                row += 1  # increment number of rows only if the study needs a subplot
                selected_subplots_studies.append(study)
            else:
                selected_first_row_studies.append(study)      
    fig = tools.make_subplots(
        rows = row,
        shared_xaxes = True,
        shared_yaxes = False,
        cols = 1,
        print_grid = True,
        vertical_spacing = 0.05,
        row_width = [0.2]*(row-1)+[1-(row-1)*0.2]
    )        
    
    fig.append_trace(globals()[type_trace](df), 1, 1)
    fig = average_close(df, index, ticker, fig)
    
    fig['layout']['xaxis1'].update(tickangle= -0, 
                             tickformat = '%Y-%m-%d',
                             autorange = True,
                             showgrid = True,
                             mirror = 'ticks',
                             color = 'rgba(255, 255, 255, 1)',
                             tickcolor = 'rgba(255,255,255,1)')
    fig['layout']['yaxis1'].update(range = [df['Close'].min()/1.3, df['Close'].max()*1.05],
                             showgrid = True,
                             anchor = 'x1', 
                             mirror = 'ticks',
                             layer = 'above traces',
                             color = 'rgba(255, 255, 255, 1)',
                             gridcolor = 'rgba(255, 255, 255, 1)'
                             )
    fig['layout']['yaxis2'].update(
#                          range= [0, df['Volume'].max()*4], 
#                          overlaying = 'y2', 
#                          layer = 'below traces',
                          autorange = True,
#                        anchor = 'x1', 
                          side = 'left', 
                          showgrid = True,
                          title = 'D. Trad. Vol.',
                          color = 'rgba(255, 255, 255, 1)',
                          )
   
    if strategyTest != '' and strategyTest is not None:
        bsSig, retStrat, retBH = globals()[strategyTest](df, start_date, end_date, 5)
        fig = bsMarkers(df, fig)
        fig['layout'].update(annotations=[
                dict(
                    x = df.index[int(round(len(df.index)/2, 0))],
                    y = df['Close'].max()*1.02,
                    text = 'Your Strategy: ' + str(round(retStrat*100, 0)) + ' % \nBuy&Hold: ' + str(round(retBH*100, 0)) + '%',
                    align = 'center',
                    font = dict(
                            size = 16,
                            color = 'rgb(255, 255, 255)'),
                            
                    showarrow = False,
                    
                )
                ]
        )

    for study in selected_first_row_studies:
        fig = globals()[study](df, fig)  # add trace(s) on fig's first row
    
    row = 2
    fig = vol_traded(df, fig, row)
    for study in selected_subplots_studies:
        row += 1
        fig = globals()[study](df, fig, row)  # plot trace on new row

    fig['layout']['margin'] = {'b': 30, 'r': 60, 'l': 40, 't': 10}    
    fig['layout'].update(
        autosize = True,
        uirevision = 'The User is always right',
        height = 600,
        paper_bgcolor = '#000000',
        plot_bgcolor = '#000000',
        hovermode = 'x',
        spikedistance = -1,
        xaxis = {'gridcolor': 'rgba(255, 255, 255, 0.3)', 'gridwidth': .5, 'showspikes': True, 'spikemode': 'across', 'spikesnap': 'data', 'spikecolor': 'rgb(220, 200, 220)', 'spikethickness': 1, 'spikedash': 'dot'},
        yaxis = {'gridcolor': 'rgba(255, 255, 255, 0.3)', 'gridwidth': .5}, 
        legend = dict(x = 1.05, y = 1)
    )
    
    fig['layout']['xaxis'].update(
        rangeselector=dict(
                buttons=list([
                        dict(count = 1,
                             label = '1m',
                             step = 'month',
                             stepmode = 'backward'),
                        dict(count = 6,
                             label = '6m',
                             step = 'month',
                             stepmode = 'backward'),
                        dict(count = 1,
                             label = 'YTD',
                             step = 'year',
                             stepmode = 'todate'),
                        dict(count = 1,
                             label = '1y',
                             step = 'year',
                             stepmode = 'backward'),
                        dict(count = 3,
                             label = '3y',
                             step = 'year',
                             stepmode = 'backward'),
                        dict(step = 'all',
                             label = 'reset (full range)')
                    ])
            ),
            rangeslider = dict(
                visible = False
            ),
            type="date"
    )
    
    return fig



def generate_info_div_callback(num):
    def info_div_callback(ticker, n_intervals, index, timeRange):
        #dh = am.DataHandler(index)
        dh = MSDataManager()
        #print(index)
        mh = MarketHours(dh.ReturnMIC(index))
        data = None
        #data = dh.ReturnDataRT(ticker)
        #children = ''
        #if not data.empty
        data = dh.ReturnData(ticker, limit=2)
        #if data is not None:
        latest = data['Close'].iloc[-1]
        latest_trade_date = dh.ReturnLatestStoredDate(ticker).strftime('%Y:%M:%d - %H:%M:%S')
        diff = data['Close'].diff()
        pct_change = data['Close'].pct_change()[-1]
        ftwh = data['Close'].values.max()
        ftwl = data['Close'].values.min()
        #market_cap = data['market_cap'].iloc[-1]
        market_cap = 0
        #curr = str(data['currency'].iloc[-1]).replace('EUR', '€')
        curr = 'USD'
        children = [html.Div([
                        html.Table([
                                html.Tr([
                                        html.Td(html.H2(ticker)),
                                        html.Td(),
                                        html.Td(),
                                        html.Td('Market cap:'),
                                        html.Td(curr + '{:0,.2f}'.format(market_cap))
                                        ]),
                                html.Tr([
                                        html.Td([html.Span(str(curr) + '%.2f'%latest + ' '), html.Span('' + '%+.2f'%diff + ' (' + '%+.2f'%pct_change + ' %)')], className = 'small'),
                                        html.Td(),
                                        html.Td(),
                                        html.Td(html.Span('Market open', style={'color': 'green'}) if mh.MarketOpen() else html.Span('Market closed', style={'color': 'red'})),
                                        html.Td()
                                        ]),
                                html.Tr([
                                        html.Td([html.Span(str(latest_trade_date))]),
                                        html.Td(),
                                        html.Td(),
                                        html.Td(),
                                        html.Td()
                                        ]),        
                                ],
                                className = 'small'
                                ),
                        ]),
                    html.Div([
                            dcc.RangeSlider(
                                    disabled = True,
                                    min=ftwl,
                                    max=ftwh,
                                    value=[ftwl, latest, ftwh],
                                    marks={
                                        ftwl: {'label': curr + str(ftwl), 'style': {'color': '#77b0b1'}},
                                        latest: {'label': curr + str(latest), 'style': {'color': '#77b0b1'}},
                                        ftwh: {'label': curr + str(ftwh), 'style': {'color': '#77b0b1'}}
                                    }
                                )                                
                            ], style={'display': 'block', 'width': '60%', 'padding': '8px 40px'})
                    ]
        return children
    return info_div_callback



def generate_figure_callback(num):
    def chart_fig_callback(ticker_list, trace_type, studies, timeRange, strategyTest, oldFig, index):
        start_date, end_date = to.axisRangeToDate(timeRange)
        
        if oldFig is None or oldFig == {"layout": {}, "data": {}}:
            return get_fig(ticker_list, trace_type, studies, strategyTest, start_date, end_date, index)
        fig = get_fig(ticker_list, trace_type, studies, strategyTest, start_date, end_date, index)
        return fig
    return chart_fig_callback

####
# Creates a list of selected stocks in hidden Div
####
app.config.suppress_callback_exceptions=True    


def generate_load_ticker_callback():
    def load_ticker_callback(selected_cells, ticker_old):
        if selected_cells:
            ticker_new = selected_cells[0]['row_id']
        else:
            ticker_new = ticker_old
        return ticker_new
    return load_ticker_callback

app.callback(
                    Output('0stock_picker', 'value'),
            [
                    Input('IndexListTable', 'selected_cells')
            ],
            [
#                    State('DAX30Table', 'selected_rows'),
#                    State('DAX30Table', 'selected_row_ids'),
                    State('0stock_picker', 'value')
            ]
            )(generate_load_ticker_callback())
#cache.memoize(timeout=100) 


def calcReturns(stockPicker, start_date, end_date, strategyToTest, index):
    dh = MSDataManager()
    df1 = dh.ReturnData(stockPicker, start_date=start_date, end_date=end_date)
    df, returnsStrat, returnsBH = globals()[strategyToTest](df1, start_date, end_date, 5)
    tickerSymbol = []
    buyPrices = []
    buyDates = []
    buySignals = []
    sellPrices = []
    sellDates = []
    sellSignals = []
    returns = []
    for i in range(len(df)):
        sig1 =df['bsSig'].iat[i]
        val1 = df['Close'].iat[i]
        date1 = df.index[i]
        signal = str(int(df['SignalMACD'][i])) + str(int(df['SignalROC'][i])) + str(int(df['SignalSTOC'][i])) + str(int(df['SignalRSI'][i]))
        if sig1 == 1:
            tickerSymbol.append(stockPicker)
            priceBuy = val1
            buyPrices.append(val1)
            buyDates.append(date1)
            buySignals.append(signal)
        elif sig1 == -1:
            priceSell = val1
            sellPrices.append(val1)
            sellDates.append(date1)
            profit = (priceSell-priceBuy)/priceBuy
            sellSignals.append(signal)
            returns.append(round(profit*100, 2))
    data = np.array([tickerSymbol, buyDates, buyPrices, buySignals, sellDates, sellPrices, sellSignals, returns]).T.tolist()
    dff = pd.DataFrame(data = data, columns = ['Symbol', 'datebuy', 'pricebuy', 'buysignals', 'datesell', 'pricesell', 'sellsignals', 'perprofit'])
    return dff, returnsStrat, returnsBH

def calcStrategy(start_date, end_date, index):
    full_market_returns = []
    full_returns_bh = []
    #dh = am.DataHandler(index)
    dh = MSDataManager()
    for ticker in dh.ReturnIndexConstituents(index):
        df1 = dh.ReturnData(ticker, start_date=start_date, end_date=end_date,)
        df, returnsStrat, returnsBH = MOM1(df1, start_date, end_date, 5)
        full_market_returns.append(returnsStrat*100)
        full_returns_bh.append(returnsBH*100)
    return [np.average(np.array(full_market_returns)), np.average(np.array(full_returns_bh))]

def generate_calc_full_market_callback():
    def calc_full_market_callback(n_clicks, timeRange, index):
#        strategyToTest = 'MOM1'
        start_date, end_date = to.axisRangeToDate(timeRange)
        children = ''
        if n_clicks != 0:
            calcStratRes= calcStrategy(start_date, end_date, index)
#            initial_vals = ([5, 12, 26, 9, 7, 5])
            children = html.Div([html.H3('%Profit Total Your Strategy: '), html.Div([str(round(calcStratRes[0], 2))]), html.H3('%Profit Total B&H: '), html.Div([str(round(calcStratRes[1], 2))])])
        return children
    return calc_full_market_callback
app.callback(       
                    Output('full-market-backtest-results', 'children'),
            [
                    Input('full_market_backtest', 'n_clicks')
            ],
            [
                    State('0chart', 'relayoutData'),
                    State('selected-index-div', 'value')
            ]
            )(generate_calc_full_market_callback())
#cache.memoize(timeout=100) 

#app.callback(
#            [
#                    Output('close_button' + str(num.replace('.', '')) + 'graph_div', 'n_clicks') for num in fse.index 
#            ],
#            [
#                    Input('charts_clicked', 'children')
#            ]
#            )(generate_reset_close_button_nclicks_callback())
#cache.memoize(timeout=100)


def generate_strategyTest_callback(num):
    def strategyTest_callback(strategyToTest, timeRange, stockPicker, index):
        """

        """

        start_date, end_date = to.axisRangeToDate(timeRange)
        res, retsum, returnsBH = calcReturns(
                stockPicker,
                start_date,
                end_date,
                strategyToTest,
                index
                )
        trace = go.Histogram(
                x=res['perprofit'],
                opacity=0.7,
                name='Proft (%)',
                marker={"line":
                        {"color": "#25232C", "width": 0.2}}, xbins={"size": 3}
                )
        layout = go.Layout(
                title="",
                width=400,
                xaxis={"title": "Profit (%)", "showgrid": False},
                yaxis={"title": "Count", "showgrid": False}
                )
        figure = {"data": [trace], "layout": layout}
        res.to_csv(path_or_buf='./Data/strategy_backtest_results.csv')
        return figure, res.to_dict('records')
    return strategyTest_callback


def generate_update_picker_dropdown():
    def update_picker_dropdown(index):
        todaydate = datetime.now()
        index = index.split('-')[1]
        #options = am.DataHandler().IndexConstituentsDict()
        dh = MSDataManager()
        #options = dh.IndexConstituentsDict(index)
        options = {'AAPL': 'AAPL', 'MSFT': 'MSFT'}
        #dh = am.DataHandler()

        data = None
        # def Ret52wData(tick):
        #     return dh.ReturnData(
        #             tick,
        #             start_date=(todaydate - relativedelta(weeks=52)),
        #             end_date=todaydate,
        #             )
        # data = [{
        #         'id': tick,
        #         'Symbol': tick,
        #         'Name': tick,
        #         '52ws': round(MomentumIndicators(Ret52wData(tick)).calcSharpe(252), 2),
        #         '52wh': Ret52wData(tick)['Close'].max(),
        #         '52wl': Ret52wData(tick)['Close'].min()
        #         } for tick in dh.ReturnIndexConstituents(index)]
        return index, options, data
    return update_picker_dropdown


for num in range(0,1):
#    num = num.re5place('.', '')
#    app.callback(
#                        Output(str(num) + 'graph_div', 'className'),
#                [
#                        Input('charts_clicked', 'children')
#                ],       
#                )(generate_show_hide_graph_div_callback(num))
#    cache.memoize(timeout=100)           
    
    


    
#    rn [{'label': i, 'value': i} for i in fnameDict[name]]    

    app.callback(
                        Output(str(num) + 'chart', 'figure'),
                [
                        Input(str(num) + 'stock_picker', 'value'),
                        Input(str(num) + 'trace_style', 'value'),
                        Input(str(num) + 'study_picker', 'value'),
                        Input(str(num) + 'chart', 'relayoutData'),
                        Input(str(num) + 'strategyTest', 'value')
                ],
                [
                        State(str(num) + 'chart', 'figure'),
                        State('selected-index-div', 'value')
                ]
                )(generate_figure_callback(num))
#    cache.memoize(timeout=100)

    app.callback(
                [
                        Output('selected-index-div', 'value'),
                        Output(str(num) + 'stock_picker', 'options'),
                        Output('IndexListTable', 'data')
                ],
                [
                        Input('stockIndexTabs', 'value')
                ]
                )(generate_update_picker_dropdown())

    app.callback(
                [
                        Output('hist-graph', 'figure'),
                        Output(str(num) + 'strategyTestResults', 'data')
                ],
                [
                        Input(str(num) + 'strategyTest', 'value')
                ],
                [
                        State(str(num) + 'chart', 'relayoutData'),
                        State(str(num) + 'stock_picker', 'value'),
                        State('selected-index-div', 'value')
                ],
                )(generate_strategyTest_callback(num))
#    cache.memoize(timeout=100)
    app.callback(
                    Output(str(num) + 'info_div', 'children'),
                    [
                            Input(str(num) + 'stock_picker', 'value'),
                            Input('interval-component', 'n_intervals'),
                    ],
                    [
                            State('selected-index-div', 'value'),
                            State('0chart', 'relayoutData')
                    ],
                )(generate_info_div_callback(num))
#    cache.memoize(timeout=100)

if __name__ == '__main__':
    app.run_server()
