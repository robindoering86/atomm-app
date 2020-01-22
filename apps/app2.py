#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Created on Tue Jul  2 09:29:34 2019

@author: robin
'''
from components import *
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as ddt
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objs as go
from plotly import tools
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime
import atomm as am
import atomm.Tools as to
from atomm.Indicators import MomentumIndicators
from atomm.DataManager.main import MSDataManager
from atomm.Tools import MarketHours
from atomm.Tools import calcIndicators
from atomm.Tools import calc_returns as calcReturns

# joblib to import trained models
from joblib import load, dump


from app import app

import colorlover as cl
colorscale = cl.scales['12']['qual']['Paired']

cgreen = 'rgb(12, 205, 24)'
cred = 'rgb(205, 12, 24)'


dh = MSDataManager()
options = dh.IndexConstituentsDict('SPY')
spy_info = pd.read_csv('./data/spy.csv')
models = [
    {'label': 'RandomForest_1', 'value': 'RF_1'},
    {'label': 'RF_trained_for_T', 'value': 'RF_trained_for_T'},
    {'label': 'RF_trained_for_IBM', 'value': 'RF_trained_for_IBM'},
    {'label': 'RF_trained_for_ABC', 'value': 'RF_trained_for_ABC'},
    {'label': 'RF_trained_for_PFG', 'value': 'RF_trained_for_PFG'},

    {'label': 'RandomForest_1_three-class_T', 'value': 'RF_1_three_class_T'},
    {'label': 'RF_two_class_thresh_T', 'value': 'RF_two_class_thresh_T'},

    ]

model_dict = {
    'RF_1': 'RF_best.joblib',
    'RF_trained_for_T': 'RF_T.joblib',
    'RF_trained_for_IBM': 'RF_IBM.joblib',
    'RF_trained_for_ABC': 'RF_ABC.joblib',
    'RF_trained_for_PFG': 'RF_PFG.joblib',

    'RF_1_three_class_T': 'RF_three_class_T.joblib',
    'RF_two_class_thresh_T': 'RF_two_class_thresh_T.joblib',
    }

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


def search_list():
    search_list = []
    for i in options:
        sym = i.get('value')
        try: 
            sec = spy_info[spy_info['Symbol'] == sym]['Security'].values[0]
        except:
            continue
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
                id='stock_picker',
                type='text',
                list='list-suggested-inputs',
                placeholder='Search for stock',
                value='MMM - 3M Company',
                className='bg-dark border-light text-light w-100 rounded-sm br-3 pl-2 pr-2 pt-1 pb-1'
                ),
            className='border-light'
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
            dbc.Col([
                html.H6('Load Pretrained Model'),
                dcc.Dropdown(
                    id='model_selector',
                    options=models,
                    multi=False,
                    value='RF_1',
                    placeholder='Select ML Model',
                    className='dash-bootstrap'
                    ),
                html.H6('Model Params'),
                dbc.Textarea(
                    id='model_params',
                    value=[],
                    placeholder='',
                    bs_size='sm',
                     style={
                         'background-color': '#222',
                         'border': 0,
                         'color': 'rgb(255, 255, 255)'
                         },
                    ),
                ],
                md=4,
                className='pl-4 pr-4 pt-2 pb-2',
                id='abc'
                ),
            dbc.Col([
                html.H6('Testing Period'),
                dcc.DatePickerRange(
                    calendar_orientation='vertical',
                    display_format='DD.MM.YYYY',
                    persistence=True,
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
                id='middle',
                className='border-left pl-4 pr-4 pt-2 pb-2',
                ),
            dbc.Col(
                children=[],
                md=4,
                className='border-left pl-4 pr-4 pt-2 pb-2',
                id='resultsbox'
                ),
            ],
            className='',
            )],
        className='border-bottom submenu',
        fluid=True,
        )
    return body
    
def create_chart_div():      
    chart = dbc.Container(
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id='returns',
                    figure={
                        'data': [],
                        'layout': chart_layout
                        },
                    config={
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
                fluid=True,
                className='content',
                )
    return chart


########
##
## MAIN DASHBOARD LAYOUT
##
########
layout = html.Div(
    [
        create_submenu(),
        create_chart_div(),
        dcc.Interval(
            id='interval-component',
            interval=30*10000,
            n_intervals=0
            ),
        html.Div(
            children=[],
            id='model_div',
            style={'display': 'none'}
            ),
        ]
    )


def return_tearsheet(df, names, start_date, end_date):

    # dh = MSDataManager()
    # df = dh.ReturnData(ticker, start_date=start_date, end_date=end_date)
   
    row=1  # number of subplots
   
    fig=tools.make_subplots(
        rows=row,
        shared_xaxes=True,
        shared_yaxes=False,
        cols=1,
        print_grid=True,
        vertical_spacing=0.05,
        row_width=[0.2]*(row-1)+[1-(row-1)*0.2]
    )
    type_trace='cum_returns'
    for name in names: 
        fig.append_trace(globals()[type_trace](df, name, name, True), 1, 1)
    df['predictions']=df['predictions'].diff().shift(-1)
    fig.append_trace(
        go.Scatter(
        x=df[df['predictions'] == 1].index,
        y=df[df['predictions'] == 1]['Cum_Returns_Strat'],
        mode='markers',
        name='Buys',
        showlegend=False,
        marker={'symbol': 'triangle-up', 'size': 15, 'color': 'green'},
        ), 1, 1)
    
    fig.append_trace(
        go.Scatter(
        x=df[df['predictions'] == -1].index,
        y=df[df['predictions'] == -1]['Cum_Returns_Strat'],
        mode='markers',
        name='Sells',
        showlegend=False,
        marker={'symbol': 'triangle-down', 'size': 15, 'color': 'red'},
        ), 1, 1)

    fig['layout'][f'xaxis{row}'].update(
        tickangle=-0, 
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
        title = 'Cumulative Returns (%)',
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
#
# Callbacks for Interactivity
#


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
                            )
                            ],
                        className='pt-0, pb-0',
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
                            )
                            ]),
                        ]
                    ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H6('Your strategy')
                        ]),
                    dbc.Col(
                        [
                            html.H6('Baseline')
                        ]),
                    ]
                ),
                ]
    return fig, textbox


@app.callback(
    Output('middle', 'children'),
    [Input('test', 'value')]
    )
def test(val): return val
