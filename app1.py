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

# cgreen = 'rgb(12, 205, 24)'
# cred = 'rgb(205, 12, 24)'


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
                value='',
                className='bg-dark text-light w-100 rounded-pill br-3 pl-2 pr-2 pt-1 pb-1'
                ),
            className=''
            ),
        dbc.Col(
            dbc.Button("Search", color="primary", className="ml-2"),
            width="auto",
        ),
    ],
    no_gutters=True,
    className="ml-auto flex-nowrap mt-3 mt-md-0",
    align="center",
)


def create_submenu():
    body = dbc.Container([
        dbc.Row(
            [
            dbc.Col(
                children=[],
                md=4,
                className='pl-4 pr-4 pt-2 pb-2',
                id='infodiv'
                ),
            dbc.Col([
                html.H6('Select Technical Indicator'),
                   
                dcc.Dropdown(id = '0study_picker',
                              options = indicators,
                              multi = True,
                              value = [],
                              placeholder = 'Select studies',
                              className = 'dash-bootstrap'
                              ),
                    
                ],
                md=4,
                className='border-left pl-4 pr-4 pt-2 pb-2',
                ),
            
            dbc.Col([
                html.H6('Chart Style'),
                dbc.RadioItems(
                            id = '0trace_style',
                            options=[
                                {'label': 'Line Chart', 'value': 'd_close'},
                                {'label': 'Candlestick', 'value': 'ohlc'}
                            ],
                            value='ohlc',
                            className='inline-block',
                        ),                    
                ],
                md=4,
                className='border-left pl-4 pr-4 pt-2 pb-2',
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
                    id = '0chart',
                    figure = {
                        'data': [],
                        'layout': [{'height': 750}]
                        },
                    config = {'displayModeBar': False, 'scrollZoom': True, 'fillFrame': False},
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
            dbc.NavbarToggler(id='navbar-toggler'),
            dbc.Collapse(
                search_bar,
                id="navbar-collapse",
                navbar=True
                ),
        ],
        color="dark",
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




def get_fig(ticker, type_trace, studies, start_date, end_date, index):

#    for ticker in ticker_list:
    #dh = am.DataHandler(index)
    dh = MSDataManager()
    df = dh.ReturnData(ticker, start_date=start_date, end_date=end_date)
   
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
        range = [
            df['Close'].min()/1.3,
            df['Close'].max()*1.05
        ],
        showgrid = True,
        anchor = 'x1', 
        mirror = 'ticks',
        layer = 'above traces',
        color = 'rgba(255, 255, 255, 1)',
        gridcolor = 'rgba(255, 255, 255, 1)'
        )
    fig['layout']['yaxis2'].update(
        # range= [0, df['Volume'].max()*4], 
        # overlaying = 'y2', 
        # layer = 'below traces',
        autorange = True,
        # anchor = 'x1', 
        side = 'left', 
        showgrid = True,
        title = 'D. Trad. Vol.',
        color = 'rgba(255, 255, 255, 1)',
        )
   
    # if model != '' and model is not None:
    #     # bsSig, retStrat, retBH = globals()[strategyTest](df, start_date, end_date, 5)
    #     ml = load(model)
    #     fig = bsMarkers(df, fig)
    #     fig['layout'].update(annotations=[
    #             dict(
    #                 x = df.index[int(round(len(df.index)/2, 0))],
    #                 y = df['Close'].max()*1.02,
    #                 text = 'Your Strategy: ' + str(round(retStrat*100, 0)) + ' % \nBuy&Hold: ' + str(round(retBH*100, 0)) + '%',
    #                 align = 'center',
    #                 font = dict(
    #                         size = 16,
    #                         color = 'rgb(255, 255, 255)'),
                            
    #                 showarrow = False,
                    
    #             )
    #             ]
    #     )

    for study in selected_first_row_studies:
        fig = globals()[study](df, fig)  # add trace(s) on fig's first row
    
    row = 2
    fig = vol_traded(df, fig, row)
    for study in selected_subplots_studies:
        row += 1
        fig = globals()[study](df, fig, row)  # plot trace on new row


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



def generate_info_div_callback():
    def info_div_callback(ticker, n_intervals):
        ticker_sym = ticker.split('-')[0].strip()
        ticker_sec = ticker.split('-')[1].strip()
        dh = MSDataManager()
        mh = MarketHours('NYSE')
        mo_class = 'cgreen' if mh.MarketOpen else 'cred'
        mo_str = 'open' if mh.MarketOpen else 'closed'
        latest_trade_date = dh.ReturnLatestStoredDate(ticker_sym)
        latest_trade_date_str = latest_trade_date.strftime('%d-%m-%Y - %H:%M:%S')
        data = dh.ReturnData(
            ticker_sym,
            start_date=latest_trade_date-relativedelta(years=1),
            end_date = latest_trade_date,
            limit=None
            )
        latest = data['Close'].iloc[-1]
        diff = (data['Close'][-2]-data['Close'][-1])
        pct_change = diff/data['Close'][-2]
        pl_class = 'cgreen' if diff > 0 else 'cred'
        pl_sign = '+' if diff > 0 else '-'
        # pct_change = f''
        ftwh = data['Close'].values.max()
        ftwl = data['Close'].values.min()
        #market_cap = data['market_cap'].iloc[-1]
        # market_cap = 0
        #curr = str(data['currency'].iloc[-1]).replace('EUR', '€')
        curr = 'USD'
        children = [
            dbc.Row(
                [
        
                        html.H6(f'{ticker_sec} ({ticker_sym})', className='')
                        
                ]
                ),
            dbc.Row(
                [
                    dbc.Col([   
                        html.Span(html.Strong(f'{curr} {latest:.2f} ', className=pl_class+' large bold')),
                        html.Span(f'{pl_sign}{diff:.2f} ', className=pl_class+' small'),
                        html.Span(f'({pl_sign}{pct_change:.2f}%)', className=pl_class+' small')
                        ],
                        md=12,
                        ),
            dbc.Row(
                [
                   dbc.Col([
        
                        html.Div([
                            html.Span(f'As of {latest_trade_date_str}. Market {mo_str}.')], className=' small'),
                            # ]
                        ],
                        md=12,
                        ),
                ])   
                ]
                ),
            # dbc.Row(
            #     html.Div([
            #         dcc.RangeSlider(
            #                 disabled = True,
            #                 min=ftwl,
            #                 max=ftwh,
            #                 value=[ftwl, latest, ftwh],
            #                 # value=[1, 20, 40]
            #                 marks={
            #                     float(ftwl): {'label': f'{curr} {ftwl:.2f}', 'style': {'color': '#77b0b1'}},
            #                     float(latest): {'label': f'{curr} {latest:.2f}', 'style': {'color': '#77b0b1'}},
            #                     float(ftwh): {'label': f'{curr} {ftwh:.2f}', 'style': {'color': '#77b0b1'}}
            #                 }
            #             )                                
            #         ],
            #         style={'display': 'block', 'width': '60%', 'padding': ''},
            #         className='mt-1'
            #         )
            #         )
            ]
        
        
        # {latest_trade_date}
            # latest}+({pct_change:.2f})')
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
        print(ticker_list)
        if oldFig is None or oldFig == {'layout': {}, 'data': {}}:
            return get_fig(ticker_list, trace_type, studies,  
                           start_date, end_date, index)
        fig = get_fig(ticker_list, trace_type, studies,  
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
    ],
    [
        State('0chart', 'figure'),
        State('selected-index-div', 'value')
    ]
    )(generate_figure_callback())

app.callback(
        Output('infodiv', 'children'),
    [
        Input('0stock_picker', 'value'),
        Input('interval-component', 'n_intervals')
    ]
    )(generate_info_div_callback())

for num in range(0,1):
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
