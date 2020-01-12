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
#import quandl as quandl 


app = dash.Dash()
theme = 'seaborn'

import colorlover as cl
colorscale = cl.scales['12']['qual']['Paired']

# Quandl for tick data
#quandl.ApiConfig.api_key = 'WfP6-Cb6XJgQJvDrLRPF'



# Set number of allowed charts
num_charts = 4

#DATA_FILE_PATH = 'C:\\Users\\ag_optik\\Nextcloud\\Coding\\atomm\\Data\\DAX\\'
DATA_FILE_PATH = './Data/DAX/'

#USERNAME_PASSWORD_PAIRS = [['Robin', 'BigShot']]
 
#auth = dash_auth.BasicAuth(app,USERNAME_PASSWORD_PAIRS)
server = app.server
 
fse = pd.read_csv('./Data/DAX_symbols.csv')
fse.set_index('Symbol', inplace=True)

options = []
for tic in fse.index:
    mydict = {}
    mydict['label'] = fse.loc[tic]['Name'] + ' (' + tic + ')'
    mydict['value'] = tic
    options.append(mydict)

indicators = [{'label': '10 day Simple moving average', 'value': 'SMA10'},
              {'label': '30 day Simple moving average', 'value': 'SMA30'},
              {'label': 'Bollinger Bands (20, 2)', 'value': 'BB202'},
              {'label': 'RSI', 'value': 'RSI'},
              {'label': 'ROC', 'value': 'ROC'},
              {'label': 'MACD(12, 26)', 'value': 'MACD'},
              {'label': 'STOC(7)', 'value': 'STOC'},
              {'label': 'ATR(7)', 'value': 'ATR'},
              {'label': 'Daily returns', 'value': 'D_RETURNS'}
              ]

def get_data(stock_ticker, timeRange):  
    df = pd.read_csv(DATA_FILE_PATH + 'OHLC-{}.csv'.format(stock_ticker), index_col = 'Date', parse_dates = True)
    start = datetime.today()
    if timeRange[-1:] == 'd':
        end = start + relativedelta(days=-int(timeRange[:-1]))
    if timeRange[-1:] == 'w':
        end = start + relativedelta(weeks=-int(timeRange[:-1]))
    elif timeRange[-1:] == 'm':
        end = start + relativedelta(months=-int(timeRange [:-1]))
    else:
        end = df.index.min()
#    df = quandl.get('FSE/{}'.format(stock_ticker), fields = 'prices', start_date=end, end_date=start)
    return df[end:start]


def create_chart_div(num):
    divs = (    html.Div(
                id = str(num) + 'graph_div',
                className = 'hidden chart-div',
                children = [
                    html.Div([
                            html.Div(
                                    children = [],   
                                    id = str(num) + 'info_div',
                                    className = 'menu_bar_left'),
                            html.Div(
                                    html.Button('', id = 'close_button' + str(num) + 'graph_div'),
                                    className = 'menu_bar_right'),
                        ],
                        className = 'menu_bar'
                            ),
                    html.Div([html.H3('Stock symbol', style = {'display': 'none'}),
                              dcc.Dropdown(id = str(num) + 'stock_picker',
                                           options = options,
                                           multi = False,
                                           value = '',
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

                    html.Div([dcc.RadioItems(id = str(num) + 'timeRangeSel',
                                             options=[
                                                     {'label': '1 W', 'value': '1w'},
                                                     {'label': '1 M', 'value': '1m'},
                                                     {'label': '3 M', 'value': '3m'},
                                                     {'label': '6 M', 'value': '6m'},
                                                     {'label': '1 Y', 'value': '12m'},
                                                     {'label': '3 Y', 'value': '36m'},
                                                     {'label': 'Max', 'value': 'max'}
                                                     ],
                                             value = '6m',
                                             labelStyle = {'display': 'inline-block'},
                                             className = 'switch-field',
                                             labelClassName = 'time-range-label',
                                             inputClassName = 'time-range-input'
                                             )
                              ],
                    style = {'display': 'inline-block'},
                    id = str(num) + 'timeRangeSel_div'
                    ),
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
                        style = {'marginTop': '0', 'textAlign': 'left', 'display': 'block'}
                    ),
                    html.Div(
                            dcc.Graph(
                                    id = str(num) + 'chart',
                                    figure = {
                                            'data': [],
                                            'layout': [{'height': 900}]
                                            },
                                    config = {'displayModeBar': False, 'scrollZoom': True, 'fillFrame': False},
#                                    style = {'width': '100vh'}
                                    ),
                            id = str(num) + 'graph',
                            )]    
                       )
            )
    return divs
########
##
## MAIN DASHBOARD LAYOUT
##
########
app.layout = html.Div([
        html.Div([
                html.H1('atomm dashboard'),
                html.Div([
                    html.Button('+', id = 'add_chart')
                    ],
                    id='slider-output-container',
                    style = {'display': 'inline-block', 'width': '20%'},
                    className = 'add_chart'
                    ),
                ]
                , className = 'header'
                ),
        html.Div([
                html.Div([
                        create_chart_div(i) for i in range(num_charts)        
                        ],
                        id = 'charts'
                        ),
                html.Div([
                        dcc.Tabs(id="tabs-with-classes",
                                 value='tab-1',
                                 parent_className='custom-tabs',
                                 className='custom-tabs-container',
                                 children=[
                                         dcc.Tab(label='DAX30', children=[
                                                html.Div([          
                                                        ddt.DataTable(
                                                            data = [{'Name': fse['Name'].loc[tick] + ' (' + tick + ')', 'Sitz': fse['Sitz (Ort)'].loc[tick], 'branche': fse['Branche'].loc[tick], '52wh': get_data(tick, '52w')['Close'].max(), '52wl': get_data(tick, '52w')['Close'].min()} for tick in fse.index],
                                                            columns = [{'id': 'Name', 'name': 'Name'}, {'id': 'Sitz', 'name': 'Sitz'}, {'id': 'branche', 'name': 'Branche'}, {'id': '52wh', 'name': '52w high'}, {'id': '52wl', 'name': '52w low'} ],
                                                            filter_action = "native",
                                                            sort_action = "native",
                                                            sort_mode = "multi",
                                                            row_selectable = "multi",
                                                            row_deletable = True,
                                                            style_data_conditional=[
                                                                {
                                                                    'if': {
                                                                        'column_id': ['52wl', '52wh'],
                                                                        'filter_query': '{52wh} - {52wl} > 0'
                                                                    },
                                                                    'backgroundColor': '#3D9970',
                                                                    'color': 'white',
                                                                }
                                                                ],
                                                        )],
                                                        id = 'table_container_div',
                                                        )                      
                                                  ],
                                                  id = 'main1',
                                                  value='tab-1',
                                                  className='custom-tab',
                                                  selected_className='custom-tab--selected'
                                                  ),
                                        dcc.Tab(label='TecDAX', children=[
                                                html.Div([]          
                                                        )                      
                                                  ],
                                                  id = 'main2',
                                                  value='tab-2',
                                                  className='custom-tab',
                                                  selected_className='custom-tab--selected'  
                                                  ),
                                        dcc.Tab(label='MDAX', children=[
                                                html.Div([]          
                                                        )                      
                                                  ],
                                                  id = 'main3',
                                                  value='tab-3',
                                                  className='custom-tab',
                                                  selected_className='custom-tab--selected'
                                                  ),
                                        dcc.Tab(label='SDAX', children=[
                                                html.Div([]          
                                                        )                      
                                                  ],
                                                  id = 'main4',
                                                  value='tab-4',
                                                  className='custom-tab',
                                                  selected_className='custom-tab--selected'
                                                  ),
                                        dcc.Tab(label='Dow Jones', children=[
                                                html.Div([]          
                                                        )                      
                                                  ],
                                                  id = 'main5',
                                                  value='tab-5',
                                                  className='custom-tab',
                                                  selected_className='custom-tab--selected'
                                                  ),
                                        dcc.Tab(label='FTSE100', children=[
                                                html.Div([]          
                                                        )                      
                                                  ],
                                                  id = 'main6',
                                                  value='tab-6',
                                                  className='custom-tab',
                                                  selected_className='custom-tab--selected'
                                                  )
                                        ]
                                )
                        ],
                        id = 'mainXX',
                        className = 'table_container'
                        ),
                ],
                id = 'main'
                ),
        html.Div(
            id = 'charts_clicked',
            style={'display': 'none'}
            ),
        html.Div(
            id = 'add_btn_counter',
            style={'display': 'none'}
            ),
        ])


####
# Plot prices, daily returns etc.
####
def d_close(df):
    trace = go.Scatter(x = df.index, y = df['Close'], mode = 'lines+markers', name = 'Daily Close', showlegend = False, line = {'width': 2})
    return trace

def ohlc(df):
    trace = go.Candlestick(x = df.index,
                           open = df['Open'],
                           high = df['High'],
                           low=df['Low'],
                           close=df['Close'],
                           showlegend = False)
    return trace

def average_close(df):
    trace = go.Scatter(x = df.index, y = (np.zeros(len(df.index)) + df['Close'].mean()), mode = 'lines', name = 'Average', showlegend = False, line = {'color': 'rgb(255, 255, 255)', 'width': 1, 'dash': 'dot'})
    return trace

def vol_traded(df):
    trace = go.Scatter(x = df.index, y = df['Volume'], showlegend = False, line = {'color': 'rgb(20, 200, 190)', 'width': 1}, fillcolor = 'rgba(20, 200, 190, 0.5)', fill = 'tozeroy', name = 'Vol. Trad.')
    return trace

def green_ref_line(df, y_pos, name):
    return (go.Scatter(x = df.index, y = (np.zeros(len(df.index)) + y_pos), name = name, showlegend = False, line = {'color': 'rgb(12, 205, 24)', 'width': 1}, hoverinfo='skip'))

def red_ref_line(df, y_pos, name):
    return (go.Scatter(x = df.index, y = (np.zeros(len(df.index)) + y_pos), name = name, showlegend = False, line = {'color': 'rgb(205, 12, 24)', 'width': 1}, hoverinfo='skip'))

####
# Studies
####
def SMA10(df, fig):
    sma10 = df['Close'].rolling(window = 12, min_periods = 1).mean()
    trace = (go.Scatter(x = df.index, y = sma10, showlegend = False, name = 'SMA10'))
    fig.append_trace(trace, 1, 1)
    return fig

def SMA30(df, fig):
    sma30 = df['Close'].rolling(window = 26, min_periods = 1).mean()
    trace = (go.Scatter(x = df.index, showlegend = False, y = sma30, name = 'SMA30'))
    fig.append_trace(trace, 1, 1)
    return fig

def BB202(df, fig):
    sma20 = df['Close'].rolling(window = 20, min_periods = 1).mean()
    std = df['Close'].rolling(window = 20, min_periods = 1).std()
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
    lower = 30
    upper = 70
    n = 5
    delta = df['Close'].diff()
    delta = delta[1:]
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up1 = up.rolling(window = n).mean()
    roll_down1 = down.rolling(window = n).mean()
    # Calculate the RSI based on EWMA
    RS1 = roll_up1 / roll_down1.abs()
    RSI1 = 100.0 - (100.0 / (1.0 + RS1))
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
    n = 5
    N = df['Close'].diff(n)
    D = df['Close'].shift(n)
    ROC = pd.Series(N / D, name='roc')    
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
    n_short = 12
    n_long = 26
    sma12 = df['Close'].rolling(window = n_short, min_periods = 1).mean()  
    sma26 = df['Close'].rolling(window = n_long, min_periods = 1).mean()
    macd = sma12-sma26
    fig.append_trace((go.Scatter(x = df.index, y = macd.where(macd >= 0).fillna(0), name = 'MACD(12,26)', mode = 'lines', connectgaps = True, showlegend = False, line = {'width': 0}, fill='tozeroy', fillcolor = 'rgba(12, 205, 24, 0.3)')), row, 1)
    fig.append_trace((go.Scatter(x = df.index, y = macd.where(macd < 0).fillna(0), name = 'MACD(12,26)', mode = 'lines', connectgaps = True, showlegend = False, line = {'width': 0}, fill='tozeroy', fillcolor = 'rgba(205, 12, 24, 0.3)')), row, 1)
    fig.append_trace((go.Scatter(x = df.index, y = macd, name = 'MACD(12,26)', mode = 'lines', showlegend = False, line = {'color': 'rgb(250, 250, 250)', 'width': 1})), row, 1)
    fig['layout']['yaxis' + str(row)].update(autorange = True,
       showgrid = True,
       title  = 'MACD(' + str(n_short) + ', ' + str(n_long) + ')',
       gridcolor = 'rgba(255, 255, 255, 0.3)',
       color = 'rgba(255, 255, 255, 1)',
       )    
    return fig

def STOC(df, fig, row):
    lower = 30
    upper = 70
    n = 7
    low = df['Close'].rolling(window = n).min()  
    high = df['Close'].rolling(window = n).max()
    latest = df['Close'].rolling(window = n).apply(lambda x: x[-1:])
    stoc = (latest-low)/(high-low)*100
    fig.append_trace(green_ref_line(df, upper, 'STOCREF' + str(upper)), row, 1)
    fig.append_trace((go.Scatter(x = df.index, y = stoc.where(stoc >= upper).fillna(upper), name = 'STOC', mode = 'lines', showlegend = False, line = {'width': 0}, fill='tonexty', fillcolor = 'rgba(12, 205, 24, 0.3)')), row, 1)
    fig.append_trace(red_ref_line(df, lower, 'STOCREF' + str(lower)), row, 1) 
    fig.append_trace((go.Scatter(x = df.index, y = stoc.where(stoc < lower).fillna(lower), name = 'STOC', mode = 'lines', showlegend = False, line = {'width': 0}, fill='tonexty', fillcolor = 'rgba(205, 12, 24, 25)')), row, 1)
    fig.append_trace((go.Scatter(x = df.index, y = stoc, name = 'STOC(7)', mode = 'lines', showlegend = False, line = {'color': 'rgb(250, 250, 250)', 'width': 1})), row, 1)

    return fig

def ATR(df, fig, row):
    n = 5
    TR=pd.Series(np.zeros(len(df),dtype=object))
    for i in range(1,len(df)):
        TR[i]=np.max([df['High'][i]-df['Low'][i], df['High'][i]-df['Close'][i-1], df['Close'][i-1]-df['Close'][i]])
    atr = TR.rolling(window = n, min_periods = 1).mean()
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
def get_fig(ticker, type_trace, studies, timeRange):

#    for ticker in ticker_list:
    df = get_data(ticker, timeRange)

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
    row = 1  # number of subplots

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
        vertical_spacing = 0.1,
    )        

    fig.append_trace(globals()[type_trace](df), 1, 1)
    fig.append_trace(average_close(df), 1, 1)
    fig.append_trace(vol_traded(df), 1, 1)
    fig['layout']['xaxis1'].update(tickangle= -0, 
                             tickformat = '%Y-%m-%d',
                             autorange = True,
                             showgrid = True,
                             mirror = 'ticks',
                             color = 'rgba(255, 255, 255, 1)',
                             tickcolor = 'rgba(255,255,255,1)')
    fig['layout']['yaxis1'].update(autorange = True,
#                                 overlaying = 'y3',
                             showgrid = True,
                             anchor = 'x1', 
                             mirror = 'ticks',
                             layer = 'above traces',
                             color = 'rgba(255, 255, 255, 1)'
                             )
    fig['layout']['yaxis' + str(row+1)]=dict(range= [0, df['Volume'].max()*2], 
                          overlaying = 'y1', 
                          layer = 'below traces',
                          anchor = 'x1', 
                          side = 'right', 
                          showgrid = False,
                          title = 'D. Trad. Vol.',
                          color = 'rgba(255, 255, 255, 1)')
    fig['data'][2].update(yaxis = 'y' + str(row+1))

    for study in selected_first_row_studies:
        fig = globals()[study](df, fig)  # add trace(s) on fig's first row
    
    row = 1
    for study in selected_subplots_studies:
        row += 1
        fig = globals()[study](df, fig, row)  # plot trace on new row

    fig['layout']['margin'] = {'b': 30, 'r': 60, 'l': 40, 't': 10}    
    fig['layout']['xaxis']['rangeslider']['visible'] = False
    fig['layout'].update(
        autosize = True,
        uirevision = 'The User is always right',
        height = 400,
        paper_bgcolor = '#000000',
        plot_bgcolor = '#000000',
        hovermode = 'x',
        spikedistance = -1,
        xaxis = {'gridcolor': 'rgba(255, 255, 255, 0.3)', 'gridwidth': .5, 'showspikes': True, 'spikemode': 'across', 'spikesnap': 'data', 'spikecolor': 'rgb(220, 200, 220)', 'spikethickness': 1, 'spikedash': 'dot'},
        yaxis = {'gridcolor': 'rgba(255, 255, 255, 0.3)', 'gridwidth': .5}, 
        legend = dict(x = 1.05, y = 1)
    )
    return fig

def replace_fig(symbol, type_trace, period, old_fig):
    fig = get_fig(symbol, type_trace, period)
#    fig['layout']['xaxis']['range'] = old_fig['layout']['xaxis']['range']  # replace zoom on xaxis, yaxis is autoscaled
    return fig

def generate_info_div_callback(num):
    def info_div_callback(ticker):
        data = get_data(ticker, '1w')['Close']
        latest = str(data[-1])
        diff = str(round(data.diff()[-1], 2))
        returns = round((data.pct_change()[-1] * 100), 2)
        ftweek = get_data(ticker, '52w')['Close']
        children = [html.Div([
                        html.Table([
                                html.Tr([
                                        html.Td(html.H3(ticker)),
                                        html.Td('52w low:'),
                                        html.Td(str(ftweek.min())),
                                        html.Td('Market cap:'),
                                        html.Td()
                                        ]),
                                html.Tr([
                                        html.Td([html.Span(str(latest) + ' EUR '), html.Span(('-' if returns < 0 else '+' + str(diff) + ' (' + str(returns) + ' %)'), className = 'small')]),
                                        html.Td('52w high:'),
                                        html.Td(str(ftweek.max())),
                                        html.Td('Index percentage:'),
                                        html.Td()
                                        ])
                                ],
                                className = 'small'
                                )
                            ])
                    ]
        return children
    return info_div_callback

def generate_show_hide_graph_div_callback(pair):
    def show_hide_graph_callback(charts_clicked):
        if charts_clicked is not None:
            charts_clicked = charts_clicked.split(',')
        else:
            charts_clicked = []
        classes = ''
        len_list = len(charts_clicked)
        if pair not in charts_clicked:
            classes = 'hidden'
        for i in range(len_list):
            if charts_clicked[i] == str(pair):
                classes = 'chart-div'
                if len_list == 1:
                    classes = classes + ' one-column'
                if len_list%2 == 0:
                        classes = classes + ' two-column'
                if len_list%3 == 0:
                        classes = classes + ' three-column'
        return classes
    return show_hide_graph_callback

#def generate_figure_config_callback(num):
#    def figure_config_callback(charts_clicked):
#        len_list = len(charts_clicked)
#        if len_list == 1:
#            classes = classes + ' one-column'
#        if len_list%2 == 0:
#                classes = classes + ' two-column'
#        if len_list%3 == 0:
#                classes = classes + ' three-column'
#        return 
        

def generate_figure_callback(num):
    def chart_fig_callback(ticker_list, trace_type, studies, timeRange, chart_list, oldFig):
        if oldFig is None or oldFig == {"layout": {}, "data": {}}:
            return get_fig(ticker_list, trace_type, studies, timeRange)
        fig = get_fig(ticker_list, trace_type, studies, timeRange)
        return fig
    return chart_fig_callback

####
# Creates a list of selected stocks in hidden Div
####
app.config.suppress_callback_exceptions=True    

def generate_chart_button_callback():
    def chart_button_callback(*args):
            pairs = ''
#            print(args)
            if args[num_charts] is not None:
                
                chart_list = args[num_charts+2]
                if chart_list is not None and chart_list != '':
                    chart_list = list(map(int, chart_list.split(',')))
                else:
                    chart_list = []
            
                input_list = [args[i] for i in range(len(args)-3)]
                if None not in input_list:
                    input_list = np.array(input_list, dtype=np.int) 
                if 1 in input_list: # One of the Close buttons was clicked
                    input_list = np.array(input_list, dtype=np.int)
                    close_index = np.argmax(input_list)
                    del chart_list[close_index]
                    print('List:'+ str(chart_list))
                else:
                    if len(chart_list) < num_charts:
                        if len(chart_list) == 0:
                            chart_list = [0]
                        else:
                            set2 = set(range(chart_list[len(chart_list)-1])) - set(chart_list)
                            if len(set2) == 0:
                                chart_list.append(max(chart_list)+1)
                            else:
                                chart_list.append(set2.pop())
#                            print('List2:'+ str(chart_list))
                for i in chart_list:
                    if pairs:
                        pairs = pairs + ',' + str(i)
                    else:                
                        pairs = str(i)
                print(pairs)
                return pairs
    return chart_button_callback

#def generate_close_graph_callback(num):
#    def close_graph_callback(n_clicks, chart_list):
#            if n_clicks != 0:
#                chart_list = chart_list.split(',')
#                pairs = ''
#                num = str(num)
#                chart_list.remove(num)
#                for chart in chart_list:
#                    if pairs:
#                        pairs = pairs + ',' + chart
#                    else:                
#                        pairs = chart
#            return pairs
#    return close_graph_callback



def generate_reset_close_button_nclicks_callback():
    def reset_close_button_nclicks_callback(n_clicks):
        if n_clicks != 0:
            return 0
    return reset_close_button_nclicks_callback

app.callback(
            Output('charts_clicked', 'children'),
        
            [
                    Input('close_button' + str(num) + 'graph_div', 'n_clicks') for num in range(num_charts)
            ]+
            [
                    Input('add_btn_counter', 'children')
            ],
            [
                    State('add_btn_counter', 'children'),
                    State('charts_clicked', 'children')
            ]
        
        )(generate_chart_button_callback())


        

for num in range(num_charts):

    app.callback(
                    Output(str(num) + 'graph_div', 'className'),
            [
                    Input('charts_clicked', 'children')
            ]
            )(generate_show_hide_graph_div_callback(num))
    
    app.callback(
                    Output('close_button' + str(num) + 'graph_div', 'n_clicks'),
            [
                    Input('add_chart', 'n_clicks')
            ]
            )(generate_reset_close_button_nclicks_callback())
            
            
    app.callback(
                    Output(str(num) + 'chart', 'figure'),
            [
                    Input(str(num) + 'stock_picker', 'value'),
                    Input(str(num) + 'trace_style', 'value'),
                    Input(str(num) + 'study_picker', 'value'),
                    Input(str(num) + 'timeRangeSel', 'value'),
                    Input('charts_clicked', 'children')
            ],
            [
                    State(str(num) + 'chart', 'figure')
            ]
            )(generate_figure_callback(num))
    
    app.callback(
                    Output(str(num) + 'info_div', 'children'),
            [
                    Input(str(num) + 'stock_picker', 'value')
            ]
            )(generate_info_div_callback(num))

 
@app.callback(
            Output('add_btn_counter', 'children'),
        [
            Input('add_chart', 'n_clicks')        
        ])
def count_add_btn_clicks(n_clicks):
    if n_clicks != 0:
        return n_clicks-1
#    app.callback(
#                    Output(str(num) + 'chart', 'config'),
#            [
#                    Input('charts_clicked', 'children')
#            ]
#            )(generate_figure_config_callback(num))
    



    #layout = go.Layout())       

#    fig = {
#        'data': traces,
#        'layout': {'title': ', '.join(stock_ticker)+' Closing Prices'}
#    }
            
#            ','.join(stock_ticker)




if __name__ == '__main__':
    app.run_server()