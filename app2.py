#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Created on Tue Jul  2 09:29:34 2019

@author: robin
'''
from components import *


import dash
#import dash_auth
import dash_core_components as dcc
import dash_html_components as html
import dash_table as ddt
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objs as go
from plotly import tools
#import plotly.io as pio
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime
#from flask_cachi1ng import Cache
import atomm as am
import atomm.Tools as to
from atomm.Indicators import MomentumIndicators
from atomm.DataManager.main import MSDataManager
from atomm.Tools import MarketHours
from atomm.Tools import calcIndicators
from atomm.Tools import calc_returns as calcReturns

# joblib to import trained models
from joblib import load, dump
#to = am.Tools()
#import time

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
theme = 'seaborn'

app.title = 'atomm web app'

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
spy_info = pd.read_csv('./data/spy.csv')
# options = [{'label': 'AAPL', 'value': 'AAPL'}, {'label': 'MSFT', 'value': 'TT'}]
indicators = [
                {'label': '10 day exponetial moving average', 'value': 'EMA10', 'subplot': False},
                {'label': '30 day exponetial moving average', 'value': 'EMA30', 'subplot': False},
                {'label': '10 day Simple moving average', 'value': 'SMA10', 'subplot': False},
                {'label': '30 day Simple moving average', 'value': 'SMA30', 'subplot': False},
                {'label': 'Bollinger Bands (20, 2)', 'value': 'BB202', 'subplot': False},
                {'label': 'RSI', 'value': 'RSI', 'subplot': True},
                {'label': 'ROC', 'value': 'ROC', 'subplot': True},
                {'label': 'MACD(12, 26)', 'value': 'MACD', 'subplot': True},
                {'label': 'STOC(7)', 'value': 'STOC', 'subplot': True},
                {'label': 'ATR(7)', 'value': 'ATR', 'subplot': True},
                {'label': 'Daily returns', 'value': 'D_RETURNS', 'subplot': False},
                {'label': 'Average Directional Indicator', 'value': 'ADX', 'subplot': True},
                {'label': 'Williams R (15)', 'value': 'WR', 'subplot': True},
              ]

models = [
    {'label': 'RandomForest_1', 'value': 'RF_1'},
    {'label': 'XGBoost_1', 'value': 'XGB_1'}  
    
    ]

model_dict = {
    'RF_1': 'RF_best.joblib'
    }

strategies = [{'label': 'Momentum follower 1', 'value': 'MOM1'}]

subplot_traces = [i.get('value') for i in indicators if i.get('subplot')]
chart_layout = {
    'height': 600,
    'margin': {'b': 30, 'r': 60, 'l': 40, 't': 10},                                    
    'plot_bgcolor': '#222222',
    'paper_bgcolor': '#222222',
    'autosize': True,
    'uirevision': 'The User is always right',
    'hovermode': 'x',
    'spikedistance': -1,
    'xaxis': {
        'gridcolor': 'rgba(255, 255, 255, 0.5)',              
        'gridwidth': .5, 'showspikes': True,
        'spikemode': 'across',
        'spikesnap': 'data',
        'spikecolor': 'rgb(220, 200, 220)',
        'spikethickness': 1, 
        'spikedash': 'dot'
        },
    'yaxis': {
        'gridcolor': 'rgba(255, 255, 255, 0.5)', 
        'gridwidth': .5
        },
    'legend': {'x': 1.05, 'y': 1}
    }



tab1_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("This is tab 1!", className="card-text"),
            dbc.Button("Click here", color="success"),
        ]
    ),
    className="mt-3",
)

tab2_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("This is tab 2!", className="card-text"),
            dbc.Button("Don't click here", color="danger"),
        ]
    ),
    className="mt-3",
)


tabs = dbc.Tabs(
    [
        dbc.Tab(
            #create_chart_div(0),
            label='Monitoring', tabClassName='w-50'
),
        dbc.Tab(tab2_content, label='Machine Learning', tabClassName='w-50'
),

    ],
)

                
def search_list():
    search_list = []
    for i in options:
        sym = i.get('value')
        try: 
            sec = spy_info[spy_info['Symbol'] == sym]['Security'].values[0]
        except:
            print(sym)
        search_list.append(f'{sym} - {sec}')
    return search_list
suggestions = search_list()
                
search_bar = dbc.Row(
    [
        html.Datalist(
            id='list-suggested-inputs', 
            children=[html.Option(value=word) for word in suggestions]),
        dbc.Col(
            dcc.Input(
                id='0stock_picker',
                type='text',
                list='list-suggested-inputs',
                placeholder='Search for stock',
                value='MMM - 3M Company',
                className='bg-dark border-light text-light w-100 rounded-sm br-3 pl-2 pr-2 pt-1 pb-1'
                ),
            className='border-light'
            ),
        # dbc.Col(
        #     dbc.Button("Search", color="primary", className="ml-2"),
        #     width="auto",
        # ),
    ],
    no_gutters=True,
    className="ml-auto flex-nowrap mt-3 mt-md-0",
    align="center",
)

def create_submenu():
    body = dbc.Container([
        dbc.Row(
            [
            dbc.Col([
                html.H6('Load Pretrained Model'),
                dcc.Dropdown(id = 'model_selector',
                              options = models,
                              multi = False,
                              value = 'RF_1',
                              placeholder = 'Select ML Model',
                              className = 'dash-bootstrap'
                              ),
                html.H6('Model Params'),
                dbc.Textarea(
                    id = 'model_params',
                    value = [],
                    placeholder = '',
                    bs_size = 'sm',
                     style = { 
                         'background-color': '#222',
                         'border': 0, 
                         'color': 'rgb(255, 255, 255)'
                         },
                    # className='bg-dark'
                    ),
                ],
                md=4,
                className='pl-4 pr-4 pt-2 pb-2',
                id='abc'
                ),
            dbc.Col([
                html.H6('Testing Period'),
                dcc.DatePickerRange(
                    calendar_orientation = 'vertical',
                    display_format = 'DD.MM.YYYY',
                    persistence = True,
                    id='testing_range',
                    max_date_allowed=datetime.today(),
                    className='dash-bootstrap',
                    style={'background-color': '#000'}
                    ),
                dbc.Button( 
                        'Run',
                        n_clicks=0,
                        id='run_backtest',
                        className='btn-primary',
                        )
 
                ],
                md=4,
                className='border-left pl-4 pr-4 pt-2 pb-2',
                ),
            dbc.Col(
                children = [],
                md = 4 ,
                className = 'border-left pl-4 pr-4 pt-2 pb-2',
                id = 'resultsbox'
                ),
            ],
            className='',
            )],
        className='border-bottom submenu',
        fluid=True,
        )
    return body
    
def create_chart_div(num):      
    chart = dbc.Container(
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id = 'returns',
                    figure = {
                        'data': [],
                        'layout': chart_layout
                        },
                    config = {
                        'displayModeBar': False,
                        'scrollZoom': False,
                        'fillFrame': False
                        },
                    )            
                ],
                md=12,
                className='m-0',
                ),
                ]),
                fluid=True ,
                className='content',
                )
    return chart

def create_navbar():
    navbar = dbc.Navbar(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col([
                            html.Img(src=app.get_asset_url('logo.png'), height='35px'),
                            dbc.NavbarBrand([html.P(html.H5('atomm'))], className='ml-1')                               
                                 ]),
  
                    ],
                    align='center',
                    no_gutters=True,
                ),
                href='#',
            ),
            dbc.Collapse(
                search_bar,
                id="navbar-collapse",
                navbar=True
                ),
            html.A(dbc.NavLink('Overview', href="#")),
            html.A(dbc.NavLink('Machine Learning', href="#")),
            dbc.NavbarToggler(id='navbar-toggler'),

        ],
        color="dark",
        sticky = 'top',
        dark=True,
        className='navbar-dark ml-auto mr-auto shadow'
    )
    return navbar


########
##
## MAIN DASHBOARD LAYOUT
##
########
app.layout = html.Div([
    create_navbar(),
    create_submenu(),
    create_chart_div(0),
    dcc.Interval(
                    id='interval-component',
                    interval=30*10000,
                    n_intervals=0
                    ),
    # html.Div([
    #     # html.Div(
    #     #     [
                
    #     #     ],
    #     #     className='content_header'
    #     #     ),
    #         dcc.Interval(
    #                 id='interval-component',
    #                 interval=30*10000,
    #                 n_intervals=0
    #                 ),
    #         html.Div(
    #             [
    #                 create_chart_div(0)
    #             ],
    #             id='charts'
    #             ),
    #         ],
    #         id='content'
    #         ),
    html.Div(
        id='charts_clicked',
        style={'display': 'none'}
        ),
    html.Div(
        children=[],
        id='model_div',
        style={'display': 'none'}
        ),
    html.Div(
        children=[],
        id='div-out',
        style={'display': 'inline-block'}
        ),
    html.Div(
        id='add_btn_counter',
        style={'display': 'none'}
        ),
        ])




def return_tearsheet(df, names, start_date, end_date):

    # dh = MSDataManager()
    # df = dh.ReturnData(ticker, start_date=start_date, end_date=end_date)
   
    row = 1  # number of subplots
   
    fig = tools.make_subplots(
        rows = row,
        shared_xaxes = True,
        shared_yaxes = False,
        cols = 1,
        print_grid = True,
        vertical_spacing = 0.05,
        row_width = [0.2]*(row-1)+[1-(row-1)*0.2]
    )
    type_trace = 'cum_returns'
    for name in names: 
        fig.append_trace(globals()[type_trace](df, name, name, True), 1, 1)
    df['predictions'] = df['predictions'].diff().shift(-1)
    fig.append_trace(
        go.Scatter(
        x = df[df['predictions'] == 1].index,
        y = df[df['predictions'] == 1]['Cum_Returns_Strat'],
        mode = 'markers',
        name = 'Buys',
        showlegend = False,
        marker = {'symbol': 'triangle-up', 'size': 15, 'color': 'green'},
        ), 1, 1)
    
    fig.append_trace(
        go.Scatter(
        x = df[df['predictions'] == -1].index,
        y = df[df['predictions'] == -1]['Cum_Returns_Strat'],
        mode = 'markers',
        name = 'Sells',
        showlegend = False,
        marker = {'symbol': 'triangle-down', 'size': 15, 'color': 'red'},
        ), 1, 1)
        
    fig['layout'][f'xaxis{row}'].update(
        tickangle= -0, 
        tickformat = '%Y-%m-%d',
        autorange = True,
        showgrid = True,
        mirror = 'ticks',
        color = 'rgba(255, 255, 255, 1)',
        tickcolor = 'rgba(255,255,255,1)'
        )
    fig['layout']['yaxis1'].update(
        autorange = True,
        showgrid = True,
        tickformat = '%',
        anchor = 'x1', 
        mirror = 'ticks',
        layer = 'above traces',
        color = 'rgba(255, 255, 255, 1)',
        gridcolor = 'rgba(255, 255, 255, 1)'
        )
    # fig['layout']['yaxis2'].update(
    #     # range= [0, df['Volume'].max()*4], 
    #     # overlaying = 'y2', 
    #     # layer = 'below traces',
    #     autorange = True,
    #     # anchor = 'x1', 
    #     side = 'left', 
    #     showgrid = True,
    #     title = 'D. Trad. Vol.',
    #     color = 'rgba(255, 255, 255, 1)',
    #     )

    fig['layout'].update(chart_layout)
    fig['layout']['xaxis'].update(
        rangeselector=dict(
                buttons=list([
                        dict(count = 1,
                              label = '1M',
                              step = 'month',
                              stepmode = 'backward'),
                        dict(count = 6,
                              label = '6M',
                              step = 'month',
                              stepmode = 'backward'),
                        dict(count = 1,
                              label = 'YTD',
                              step = 'year',
                              stepmode = 'todate'),
                        dict(count = 1,
                              label = '1Y',
                              step = 'year',
                              stepmode = 'backward'),
                        dict(count = 3,
                              label = '3Y',
                              step = 'year',
                              stepmode = 'backward'),
                        dict(step = 'all',
                              label = 'MAX')
                    ])
            ),
            rangeslider = dict(
                visible = False
            ),
            type="date"
    )

    return fig



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


# def calcReturns(stockPicker, start_date, end_date, strategyToTest, index):
#     dh = MSDataManager()
#     df1 = dh.ReturnData(stockPicker, start_date=start_date, end_date=end_date)
#     df, returnsStrat, returnsBH = globals()[strategyToTest](
#         df1,
#         start_date,
#         end_date, 
#         5
#         )
#     tickerSymbol = []
#     buyPrices = []
#     buyDates = []
#     buySignals = []
#     sellPrices = []
#     sellDates = []
#     sellSignals = []
#     returns = []
#     for i in range(len(df)):
#         sig1 =df['bsSig'].iat[i]
#         val1 = df['Close'].iat[i]
#         date1 = df.index[i]
#         signal = str(int(df['SignalMACD'][i])) + 
# str(int(df['SignalROC'][i])) + str(int(df['SignalSTOC'][i])) + str(int(df['SignalRSI'][i]))
#         if sig1 == 1:
#             tickerSymbol.append(stockPicker)
#             priceBuy = val1
#             buyPrices.append(val1)
#             buyDates.append(date1)
#             buySignals.append(signal)
#         elif sig1 == -1:
#             priceSell = val1
#             sellPrices.append(val1)
#             sellDates.append(date1)
#             profit = (priceSell-priceBuy)/priceBuy
#             sellSignals.append(signal)
#             returns.append(round(profit*100, 2))
#     data = np.array([tickerSymbol, buyDates, buyPrices, buySignals, sellDates,
#                      sellPrices, sellSignals, returns]).T.tolist()
#     dff = pd.DataFrame(data = data, columns = ['Symbol', 'datebuy', 'pricebuy', 
#                                                'buysignals', 'datesell', 
#                                                'pricesell', 'sellsignals', 
#                                                'perprofit'])
#     return dff, returnsStrat, returnsBH

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
    return [np.average(np.array(full_market_returns)),
            np.average(np.array(full_returns_bh))]

def generate_calc_full_market_callback():
    def calc_full_market_callback(n_clicks, timeRange, index):
#        strategyToTest = 'MOM1'
        start_date, end_date = to.axisRangeToDate(timeRange)
        children = ''
        if n_clicks != 0:
            calcStratRes= calcStrategy(start_date, end_date, index)
#            initial_vals = ([5, 12, 26, 9, 7, 5])
            children = html.Div([html.H3('%Profit Total Your Strategy: '),
                                 html.Div([str(round(calcStratRes[0], 2))]),
                                 html.H3('%Profit Total B&H: '),
                                 html.Div([str(round(calcStratRes[1], 2))])])
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
        options = dh.IndexConstituentsDict(index)
        #options = {'AAPL': 'AAPL', 'MSFT': 'MSFT'}
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

def generate_figure_callback():
    def chart_fig_callback(ticker_list, trace_type, studies, timeRange, 
                           oldFig, index):
        start_date, end_date = to.axisRangeToDate(timeRange)
        ticker_list = ticker_list.split('-')[0].strip()
        if oldFig is None or oldFig == {'layout': {}, 'data': {}}:
            return get_fig(ticker_list, trace_type, studies,  
                           start_date, end_date, index)
        fig = get_fig(ticker_list, trace_type, studies,  
                      start_date, end_date, index)
        return fig
    return chart_fig_callback


#
# Callbacks for Interactivity
#

# app.callback(
                    
#         Output('0chart', 'figure'),
#     [
#         Input('0stock_picker', 'value'),
#         Input('0trace_style', 'value'),
#         Input('0study_picker', 'value'),
#         Input('0chart', 'relayoutData'),
#     ],
#     [
#         State('0chart', 'figure'),
#         State('selected-index-div', 'value')
#     ]
#     )(generate_figure_callback())

@app.callback(
    [
        Output('model_div', 'children'),
        Output('model_params', 'value'),
    ],
    [
        Input('model_selector', 'value'),
    ]
    )
def load_model_callback(model):

    if model is not None:
        model_file = model_dict.get(model)
        model_file = f'./data/{model_file}'
        model = load(model_file)
        return model_file, str(model)
    return '', 'No model selected'

@app.callback(
    [   
          Output('returns', 'figure'),
          Output('resultsbox', 'children'),
       ],
    [
          Input('run_backtest', 'n_clicks')
      ],
    [
          State('model_div', 'children'),
          State('0stock_picker', 'value'),
          State('testing_range', 'start_date'),
          State('testing_range', 'end_date'),
      ]
    )
def run_model(
        n_clicks,
        model, 
        symbol = 'AAPL',
        start_date = datetime.today()-relativedelta(years=1),
        end_date = datetime.today()
        ):
    if n_clicks > 0 and n_clicks is not None:
        start_date = datetime.strptime(start_date.split(' ')[0], '%Y-%m-%d')
        end_date = datetime.strptime(end_date.split(' ')[0], '%Y-%m-%d')
        symbol = symbol.split('-')[0].strip()
        print(symbol)
        df = dh.ReturnData(symbol, start_date=start_date, end_date=end_date)
        prices = df.copy()
        lookback_windows = [3, 5, 7, 10, 15, 20, 25, 30]
        ti_list = ['stoc', 'atr', 'williamsr', 'cci', 'rsi', 'adx', 'macd',\
           'roc', 'ema', 'bb', 'stocd']
        df = pd.concat([df], keys=[symbol], axis=1)
        X_test = calcIndicators(df, symbol, lookback_windows, ti_list)[max(lookback_windows):]
        
        model = load(model)
        y_pred = model.predict(X_test)
        results = calcReturns(y_pred, 1, prices[max(lookback_windows):])
        results['predictions'] = y_pred
        
        fig = return_tearsheet(
            results,
            ['Cum_Returns_Strat', 'Cum_Returns_Baseline'],
            start_date, 
            end_date
            )
        ret_strat = results['Cum_Returns_Strat'][-1]*100
        ret_baseline = results['Cum_Returns_Baseline'][-1]*100
        c_1 = cgreen if ret_strat > ret_baseline else cred
        c_2 = cgreen if ret_strat < ret_baseline else cred

        textbox = [
            html.H5('Results'),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Span(
                                f'{ret_strat:.2f} %',
                                style={
                                    'font-size': '2em',
                                    # 'color': 'rgb(64, 68, 72)',
                                    'color': '#fff',
                                    # 'background-color': c_1
                                    }
                            ),
                            html.Br(),
                            html.H6('Your strategy')
                        ]
                    ),
                    dbc.Col(
                        [
                            html.Span(
                                f'{ret_baseline:.2f} %',
                                style={
                                    'font-size': '2em',
                                    # 'color': 'rgb(64, 68, 72)',
                                    'color': '#fff',
                                    # 'background-color': c_1
                                    }
                            ), 
                            html.Br(),
                            html.H6('Baseline')
                        ]
                    ),
                ]
                ),
        ]
        # strat_ret = results['Cum_Returns_Strat'][-1]
        # bh_ret = results['Cum_Returns_Baseline'][-1]
    return fig, textbox


# for num in range(0,1):
#     app.callback(
#                 [
#                         Output('selected-index-div', 'value'),
#                         Output(str(num) + 'stock_picker', 'options'),
#                         Output('IndexListTable', 'data')
#                 ],
#                 [
#                         Input('stockIndexTabs', 'value')
#                 ]
#                 )(generate_update_picker_dropdown())

    # app.callback(
    #             [
    #                     Output('hist-graph', 'figure'),
    #                     Output(str(num) + 'strategyTestResults', 'data')
    #             ],
    #             [
    #                     Input(str(num) + 'strategyTest', 'value')
    #             ],
    #             [
    #                     State(str(num) + 'chart', 'relayoutData'),
    #                     State(str(num) + 'stock_picker', 'value'),
    #                     State('selected-index-div', 'value')
    #             ],
    #             )(generate_strategyTest_callback(num))
#    cache.memoize(timeout=100)
    # app.callback(
                #     Output(str(num) + 'info_div', 'children'),
                #     [
                #             Input(str(num) + 'stock_picker', 'value'),
                #             Input('interval-component', 'n_intervals'),
                #     ],
                #     [
                #             State('selected-index-div', 'value'),
                #             State('0chart', 'relayoutData')
                #     ],
                # )(generate_info_div_callback(num))
#    cache.memoize(timeout=100)

if __name__ == '__main__':
    app.run_server()
