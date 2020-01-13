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
#import quandl as quandl 
import atomm as am
import atomm.Tools as to
from atomm.Indicators import MomentumIndicators
from atomm.DataManager.main import MSDataManager
from atomm.Tools import MarketHours
#to = am.Tools()
#import time

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
theme = 'seaborn'

#cache = Cache(app.server, config={
#    'CACHE_TYPE': 'filesystem',
#    'CACHE_DIR': 'cache-directory'
#})

import colorlover as cl
colorscale = cl.scales['12']['qual']['Paired']

# cgreen = 'rgb(12, 205, 24)'
# cred = 'rgb(205, 12, 24)'


# Set number of allowed charts
num_charts = 4

server = app.server

#conf = am.Config()


#dh = am.DataHandler('DAX')
dh = MSDataManager()
options = dh.IndexConstituentsDict('SPY')
# options = [{'label': 'AAPL', 'value': 'AAPL'}, {'label': 'MSFT', 'value': 'TT'}]
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
                {'label': 'Daily returns', 'value': 'D_RETURNS'},
                {'label': 'Average Directional Indicator', 'value': 'ADX'},
                {'label': 'Williams R (15)', 'value': 'WR'},
              ]

strategies = [{'label': 'Momentum follower 1', 'value': 'MOM1'}]

def create_chart_div(num):

    body = dbc.Container([
        dbc.Row(
            dbc.Col(html.Div(html.H1('Hello World'), id='0info_div'), md=12)            
            ), 
        dbc.Row([
            dbc.Col([
                html.H6('Select stock symbol'),
                dcc.Dropdown(id = str(num) + 'stock_picker',
                                    options = options,
                                    multi = False,
                                    value = 'AAPL',
                                    placeholder = 'Enter stock symbol',
                                    className = 'dash-bootstrap'
                                    ),
                ],
                md=4),
            dbc.Col([
                html.H6('Select Technical Indicator'),
                   
                dcc.Dropdown(id = str(num) + 'study_picker',
                              options = indicators,
                              multi = True,
                              value = [],
                              placeholder = 'Select studies',
                              className = 'dash-bootstrap'
                              ),
                    
                ],
                md=4),
            
            dbc.Col([
                #html.H2('Select stock symbol'),
                dbc.RadioItems(
                            id = str(num) + 'trace_style',
                            options=[
                                {'label': 'Line Chart', 'value': 'd_close'},
                                {'label': 'Candlestick', 'value': 'ohlc'}
                            ],
                            value='ohlc',
                        ),                    
                ],
                md=4),
            dbc.Col([
                #html.H2('Select stock symbol'),                               
                dcc.Dropdown(
                    id = str(num) + 'strategyTest',
                    options = strategies,
                    multi = False,
                    value = '',
                    placeholder = 'Test strategies',
                    className = 'dash-bootstrap'
                    ),                    
                ],
                md=4),

            dbc.Col([
                #html.H2('Select stock symbol'),               
                dcc.Graph(
                        id = '0chart',
                        figure = {
                                'data': [],
                                'layout': [{'height': 900}]
                                },
                        config = {'displayModeBar': False, 'scrollZoom': True, 'fillFrame': False},
                )            
                ],
                md=12),

            ])
        ],
        className=['mt-3', 'mb-3', 'text-left']
    )
    return body



        
 




# def create_chart_div(num):
#     divs = (    dbc.Row(
#                 id = str(num) + 'graph_div',
#                 className = 'visible chart-div one-column',
#                 children = [
#                     html.Div([
#                             html.Div(
#                                     children = [html.H1('Hello World')],   
#                                     id = str(num) + 'info_div',
#                                     className = 'menu_bar_left'),
              
#                         ],
#                         className = 'menu_bar'
#                         ),
                    
                    
   
                    
                    
#                     html.Div([html.H3('Studies', style = {'display': 'none'}),
#                             dcc.Dropdown(id = str(num) + 'study_picker',
#                                       options = indicators,
#                                       multi = True,
#                                       value = [],
#                                       placeholder = 'Select studies',
#                                       className = 'dash-bootstrap'
#                         )], 
#                         style = {'display': 'inline-block', 'verticalAlign': 'top'},
#                         id = str(num) + 'study_picker_div',
#                         className = 'study_picker'
#                     ),
#                     html.Div([html.H3('Strategy', style = {'display': 'none'}),
#                             dcc.Dropdown(id = str(num) + 'strategyTest',
#                                       options = strategies,
#                                       multi = False,
#                                       value = '',
#                                       placeholder = 'Test strategies',
#                                       className = 'dash-bootstrap'
#                         )], 
#                         style = {'display': 'inline-block', 'verticalAlign': 'top'},
#                         id = str(num) + 'strategie_picker_div',
#                         className = 'strategie_picker'
#                     ),
#                     html.Div(
#                         dcc.RadioItems(
#                             id = str(num) + 'trace_style',
#                             options=[
#                                 {'label': 'Line Chart', 'value': 'd_close'},
#                                 {'label': 'Candlestick', 'value': 'ohlc'}
#                             ],
#                             value='ohlc',
#                         ),
#                         id=str(num) + 'style_tab',
#                         style = {'marginTop': '0', 'textAlign': 'left', 'display': 'inline-block'}
#                     ),
#                     html.Div(
#                             dcc.Graph(
#                                     id = str(num) + 'chart',
#                                     figure = {
#                                             'data': [],
#                                             'layout': [{'height': 900}]
#                                             },
#                                     config = {
#                                               'displayModeBar': False, 
#                                               'scrollZoom': True,
#                                               'fillFrame': False},
#                             ),
#                             id = str(num) + 'graph',
#                             )
#                             ]
#                        )
#             )
#     return divs


def create_navbar():
    navbar = dbc.NavbarSimple(
        children=[
            html.Div(id='logo', className=['logo']),
            dbc.NavItem(dbc.NavLink("Page 1", href="#")),
            dbc.DropdownMenu(
                children=[
                    dbc.DropdownMenuItem("More pages", header=True),
                    dbc.DropdownMenuItem("Page 2", href="#"),
                    dbc.DropdownMenuItem("Page 3", href="#"),
                ],
                nav=True,
                in_navbar=True,
                label="More",
            ),
        ],
        brand="web app",
        brand_href="#",
        color="dark",
        dark=True,
        className='navbar-dark',
    )
    return navbar



########
##
## MAIN DASHBOARD LAYOUT
##
########
app.layout = html.Div([
        html.Div([
                # html.Div(
                #         [
                #                 html.H1('atomm trading app'),
                #         ],
                #         className='content_header'
                #         ),
                html.Div(
                        [
                                create_navbar(),
                        ],
                        className='content_header'
                        ),
                dcc.Interval(
                        id='interval-component',
                        interval=30*10000,
                        n_intervals=0
                        ),
                html.Div(
                        [
                                create_chart_div(0)
                        ],
                        id='charts'
                        ),
                # html.Div(
                #         [
                #                 create_strategyTestResult(0)
                #         ]
                #         ),
                #html.Div(
                #        [
                #                create_stockList(0)
                #        ]
                #        )
                ],
                id='content'
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
        'ATR',
        'ADX',
        'WR'
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
    fig['layout']['yaxis1'].update(range = [
        df['Close'].min()/1.3,
        df['Close'].max()*1.05],
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
        # paper_bgcolor = '#000000',
        paper_bgcolor = '#222',

        # paper_bgcolor = 'rgb(64, 68, 72)',
        # paper_bgcolor = 'rgb(219, 240, 238)',

        # plot_bgcolor = '#000000',
        plot_bgcolor = '#222',

        # plot_bgcolor = 'rgb(64, 68, 72)',
        # plot_bgcolor = 'rgb(219, 240, 238)',

        hovermode = 'x',
        spikedistance = -1,
        xaxis = {
            'gridcolor': 'rgba(255, 255, 255, 0.5)',              
            'gridwidth': .5, 'showspikes': True,
            'spikemode': 'across',
            'spikesnap': 'data',
            'spikecolor': 'rgb(220, 200, 220)',
            'spikethickness': 1, 'spikedash': 'dot'
            },
        yaxis = {'gridcolor': 'rgba(255, 255, 255, 0.5)', 'gridwidth': .5}, 
        legend = dict(x = 1.05, y = 1)
    )
    
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



def generate_info_div_callback(num):
    def info_div_callback(ticker, n_intervals, index, timeRange):
        #dh = am.DataHandler(index)
        dh = MSDataManager()
        #print(index)
        mh = MarketHours('NYSE')
        data = None
        #data = dh.ReturnDataRT(ticker)
        #children = ''
        #if not data.empty
        data = dh.ReturnData(ticker, limit=2)
        #if data is not None:
        latest = data['Close'].iloc[-1]
        latest_trade_date = dh.ReturnLatestStoredDate(ticker).strftime('%Y:%M:%d - %H:%M:%S')
        diff = data['Close'].diff()[-1]
        pct_change = data['Close'].pct_change()[-1]
        ftwh = data['Close'].values.max()
        ftwl = data['Close'].values.min()
        #market_cap = data['market_cap'].iloc[-1]
        market_cap = 0
        #curr = str(data['currency'].iloc[-1]).replace('EUR', '€')
        curr = 'USD'
        children = [html.Div([
                        html.Table([
                                # html.Tr([
                                #         html.Td(html.H2(ticker)),
                                #         html.Td(),
                                #         html.Td(),
                                #         html.Td('Market cap:'),
                                #         html.Td(f'{curr} {market_cap:.2f}')
                                #         ]),
                                # html.Tr([
                                #         html.Td([
                                #             html.Span(f'{curr}{latest:.2f}'),
                                #             html.Span(f'{diff:.2f} + ({pct_change:.2f})')
                                #             ], className = 'small'),
                                #         html.Td(),
                                #         html.Td(),
                                #         html.Td(html.Span('Market open', style={'color': 'green'}) if mh.MarketOpen() else html.Span('Market closed', style={'color': 'red'})),
                                #         html.Td()
                                #         ]),
                                # html.Tr([
                                #         html.Td([html.Span(str(latest_trade_date))]),
                                #         html.Td(),
                                #         html.Td(),
                                #         html.Td(),
                                #         html.Td()
                                #         ]),        
                                ],
                                className = 'small'
                                ),
                        ]),
                    # html.Div([
                    #         dcc.RangeSlider(
                    #                 disabled = True,
                    #                 min=ftwl,
                    #                 max=ftwh,
                    #                 value=[ftwl, latest, ftwh],
                    #                 marks={
                    #                     ftwl: {'label': curr + str(ftwl), 'style': {'color': '#77b0b1'}},
                    #                     latest: {'label': curr + str(latest), 'style': {'color': '#77b0b1'}},
                    #                     ftwh: {'label': curr + str(ftwh), 'style': {'color': '#77b0b1'}}
                    #                 }
                    #             )                                
                    #         ], style={'display': 'block', 'width': '60%', 'padding': '8px 40px'})
                    ]
        # children = html.H1(ticker)
        return children
    return info_div_callback





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
    df, returnsStrat, returnsBH = globals()[strategyToTest](
        df1,
        start_date,
        end_date, 
        5
        )
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
    data = np.array([tickerSymbol, buyDates, buyPrices, buySignals, sellDates,
                     sellPrices, sellSignals, returns]).T.tolist()
    dff = pd.DataFrame(data = data, columns = ['Symbol', 'datebuy', 'pricebuy', 
                                               'buysignals', 'datesell', 
                                               'pricesell', 'sellsignals', 
                                               'perprofit'])
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
                           strategyTest, oldFig, index):
        start_date, end_date = to.axisRangeToDate(timeRange)
        # print(ticker_list, trace_type, studies, timeRange, strategyTest, oldFig, index)
        if oldFig is None or oldFig == {"layout": {}, "data": {}}:
            return get_fig(ticker_list, trace_type, studies, strategyTest, 
                           start_date, end_date, index)
        fig = get_fig(ticker_list, trace_type, studies, strategyTest, 
                      start_date, end_date, index)
        return fig
    return chart_fig_callback

app.callback(
                    
        Output('0chart', 'figure'),
    [
        Input('0stock_picker', 'value'),
        Input('0trace_style', 'value'),
        Input('0study_picker', 'value'),
        Input('0chart', 'relayoutData'),
        Input('0strategyTest', 'value')
    ],
    [
        State('0chart', 'figure'),
        State('selected-index-div', 'value')
    ]
    )(generate_figure_callback())


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
