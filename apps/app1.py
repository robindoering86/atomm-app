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
from app import app


import colorlover as cl
colorscale = cl.scales['12']['qual']['Paired']

cgreen = 'rgb(12, 205, 24)'
cred = 'rgb(205, 12, 24)'

dh = MSDataManager()
options = dh.IndexConstituentsDict('SPY')
spy_info = pd.read_csv('./data/spy.csv')

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


dh = MSDataManager()
options = dh.IndexConstituentsDict('SPY')
spy_info = pd.read_csv('./data/spy.csv')
config = {}
config['indicators'] = [
    {'selector':
     {'label': '10 day exponetial moving average', 'value': 'EMA10'},
     'subplot': False
     },
    {'selector':
     {'label': '30 day exponetial moving average', 'value': 'EMA30'},
         'subplot': False
     },
    {'selector':
        {'label': '10 day Simple moving average', 'value': 'SMA10'},
        'subplot': False
     },
    {'selector':
        {'label': '30 day Simple moving average', 'value': 'SMA30'},
        'subplot': False
     },
    {'selector':
        {'label': 'Bollinger Bands (20, 2)', 'value': 'BB202'},
        'subplot': False
     },
    {'selector':
         {'label': 'RSI', 'value': 'RSI'},
         'subplot': True
     },
    {'selector':
         {'label': 'ROC', 'value': 'ROC'},
         'subplot': True
     },
    {'selector':
         {'label': 'MACD(12, 26)', 'value': 'MACD'},
         'subplot': True
     },
    {'selector':
         {'label': 'STOC(7)', 'value': 'STOC'},
         'subplot': True
     },
    {'selector':
         {'label': 'ATR(7)', 'value': 'ATR'},
         'subplot': True
     },
    {'selector':
         {'label': 'Daily returns', 'value': 'D_RETURNS'},
         'subplot': True
     },
    {'selector':
         {'label': 'Average Directional Indicator', 'value': 'ADX'},
         'subplot': True
     },
    {
     'selector':
         {'label': 'Williams R (15)', 'value': 'WR'},
     'subplot': True
     },
    ]

subplot_traces = [i.get('selector').get('value') for i in config.get('indicators') if i.get('subplot')]
indicators = [i.get('selector') for i in config.get('indicators')]
chart_layout = {
    #'height': '100vh',
    'margin': {'b': 10, 'r': 20, 'l': 0, 't': 10},
    'plot_bgcolor': '#131722',
    'paper_bgcolor': '#131722',
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
        'spikedash': 'dot',
        'showline': True,
        },
    'yaxis': {
        'gridcolor': 'rgba(255, 255, 255, 0.5)',
        'gridwidth': .5,
        'showline': True,
        },
    'legend': {'x': 1.05, 'y': 1}
    }
def build_modal():
    modal = []
    for ind in indicators:
        btn = dbc.Button(
            f'{ind["label"]}',
            id=f'{ind["value"]}_collapse_button',
            className="mb-1 w-100 text-left",
            color='dark',
            size='sm',
            )
        modal.append(btn)
        collapse = dbc.Collapse(
            [
                dbc.InputGroup(
                [
                    dbc.InputGroupAddon('Set window length (default=30)', addon_type='prepend'),
                    dbc.Input(type='number', step=1, value=30, bs_size='sm', id=f'{ind["value"]}_win'),
                    dbc.InputGroupAddon(
                        dbc.Button('Set', id=f'set_{ind["value"]}', size='sm', className='')
                    ),
                ],
                size='sm',
                )
            ],
                id=f'{ind["value"]}_collapse',
            )
        modal.append(collapse)
    return modal

def create_indicator_modal():
    indicator_modal = dbc.Modal(
                    [
                        dbc.ModalHeader(
                        [
                            'Indicators',
                            dbc.Button(
                                'Close',
                                id='close_indicator_modal',
                                className='ml-auto'
                                )
                        ]
                        ),
                        dbc.ModalBody(
                            build_modal()
                        ),
                        #dbc.ModalFooter(

                        #),
                    ],
                    id='indicator_modal',
                    size='lg',
                    centered=True,
                    contentClassName='modal-dark',
                    scrollable=True,
                    backdrop=False,
                    )
    return indicator_modal


def create_submenu():
    body = dbc.Container([
        # dbc.Row(
        #     [
        #     dbc.Col(
        #         children=[],
        #         md=4,
        #         className='pl-4 pr-4 pt-2 pb-2',
        #         id='infodiv'
        #         ),
        #     dbc.Col([
        #         html.H6('Select Technical Indicator'),
        #
        #         dcc.Dropdown(id = '0study_picker',
        #                       options = indicators,
        #                       multi = True,
        #                       value = [],
        #                       placeholder = 'Select studies',
        #                       className = 'dash-bootstrap'
        #                       ),
        #
        #         ],
        #         md=4,
        #         className='border-left pl-4 pr-4 pt-2 pb-2',
        #         ),
        #
        #     dbc.Col([
        #         html.H6('Chart Style'),
        #
        #         ],
        #         md=4,
        #         className='border-left pl-4 pr-4 pt-2 pb-2',
        #         ),
        #     ],
        #     className='',
        #     ),
        dbc.Row(
        [
            dbc.Nav(
            [
                dbc.NavItem(dbc.NavLink([html.Span([html.I('settings', className='material-icons')])], href="#"), id='settings_but', className='border-right'),
                dbc.Tooltip(
                    'Settings',
                    target='settings_but',
                    ),
                #dbc.NavItem(dbc.NavLink([html.Span([html.I('show_chart', className='material-icons')])], href="#"), id='chart_style_but', className='border-right'),
                dbc.Tooltip(
                    'Select Chart Style.',
                    target='chart_style_but',
                    ),
                dbc.NavItem(
                    dbc.Button(
                                #'trending_up',
                        'Indicators',
                        id='open_indicator_modal',
                        color='link',
                    ),
                    className='border-right'
                ),
                create_indicator_modal(),
                html.Datalist(
                    id='list-suggested-inputs',
                    children=[html.Option(value=word) for word in suggestions]
                ),
                dbc.NavItem(
                    dbc.Input(
                        id='stock_picker',
                        list='list-suggested-inputs',
                        placeholder='Search for stock',
                        value='MMM - 3M Company',
                        type='text',
                        size='30',
                        className='bg-dark border-0 text-light pl-2 pr-2 pt-1 pb-1'
                        ),
                #],
                        className='border-right'
                        ),
                dbc.NavItem(
                    dbc.DropdownMenu(
                        [
                            dbc.DropdownMenuItem('Candles', id='btn_style_candles', active=True),
                            dbc.DropdownMenuItem('Line', id='btn_style_line', active=False),
                        ],
                        nav=True,
                        label='show_chart',
                        caret=False,
                        color='link',
                        id='chart_style_but',
                        className=''
                    ),
                    id='test',
                    className='border-right'
                    ),
                dbc.Tooltip(
                    'Select Chart Style.',
                    target='chart_style_but',
                    ),

                ],
                pills=False,
                )
        ]
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
                    id = '0chart',
                    figure = {
                        'data': [],
                        'layout': chart_layout
                        },
                    config = {
                        'displayModeBar': False,
                        'scrollZoom': False,
                        'fillFrame': False
                        },
                    style={'height': '85vh'}
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




########
##
## MAIN DASHBOARD LAYOUT
##
########
layout = html.Div([
    # create_navbar(),
    create_submenu(),
    create_chart_div(),
    dcc.Interval(
        id='interval-component',
        interval=30*10000,
        n_intervals=0
        ),
    html.Div(
        id='symbol_selected',
        style={'display': 'none'}
        ),
    html.Div(
        id='chart_style_div',
        style={'display': 'none'}
    ),
    html.Div(
        id='selected_study_div',
        style={'display': 'none'}
    ),
        ])


def get_fig(ticker, type_trace, studies, start_date, end_date):
    if ticker is not (None or ''):
        dh = MSDataManager()
        df = dh.ReturnData(ticker, start_date=start_date, end_date=end_date)

    selected_subplots_studies = []
    selected_first_row_studies = []
    row = 2  # number of subplots
    if studies is None:
        studies = []
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
        tickcolor = 'rgba(255, 255, 255, 1)'
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
        autorange = True,
        side = 'left',
        showgrid = True,
        title = 'D. Trad. Vol.',
        color = 'rgba(255, 255, 255, 1)',
        )

    for study in selected_first_row_studies:
        fig = globals()[study['name']](df, fig, **study['args'])  # add trace(s) on fig's first row

    row = 2
    fig = vol_traded(df, fig, row)
    for study in selected_subplots_studies:
        row += 1
        fig = globals()[study['name']](df, fig, row, **study['args'])  # plot trace on new row

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
# @app.callback(
#         Output('chart_style_div', 'children'),
#     [
#         Input('chart_style_candle', 'n_clicks'),
#         Input('chart_style_line', 'n_clicks')
#     ]
# )
# def set_chart_style(n1, n2):
#     if n1:
#         return 'ohlc'
#     if n2:
#         return 'd_close'

@app.callback(
    [
        Output('chart_style_div', 'children'),
        Output('btn_style_candles', 'active'),
        Output('btn_style_line', 'active'),
    ],
    [
        Input('btn_style_candles', 'n_clicks'),
        Input('btn_style_line', 'n_clicks'),
    ],
)
def toggle_buttons(candles, line):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if not any([candles, line,]):
        return 'ohlc', False, False
    elif button_id == 'btn_style_candles':
        return 'ohlc', True, False
    elif button_id == 'btn_style_line':
        return 'd_close', False, True


@app.callback(
        Output('0chart', 'figure'),
    [
        Input('symbol_selected', 'value'),
        Input('chart_style_div', 'children'),
        #Input('0study_picker', 'value'),
        Input('selected_study_div', 'children'),
        Input('0chart', 'relayoutData'),
    ],
    [
        State('0chart', 'figure'),
    ]
    )
def chart_fig_callback(ticker_list, trace_type, studies, timeRange,
                           oldFig):
    start_date, end_date = to.axisRangeToDate(timeRange)
    ticker_list = ticker_list.split('-')[0].strip()
    if oldFig is None or oldFig == {'layout': {}, 'data': {}}:
        return get_fig(ticker_list, trace_type, studies,
                       start_date, end_date)
    fig = get_fig(ticker_list, trace_type, studies,
                  start_date, end_date)
    app.title = (f'{ticker_list}')
    return fig

# @app.callback(
#         Output('infodiv', 'children'),
#     [
#         Input('symbol_selected', 'value'),
#         Input('interval-component', 'n_intervals')
#     ]
#     )
# def info_div_callback(ticker, n_intervals):
#     ticker_sym = ticker.split('-')[0].strip()
#     ticker_sec = ticker.split('-')[1].strip()
#     dh = MSDataManager()
#     mh = MarketHours('NYSE')
#     mo_class = 'cgreen' if mh.MarketOpen else 'cred'
#     mo_str = 'open' if mh.MarketOpen else 'closed'
#     latest_trade_date = dh.ReturnLatestStoredDate(ticker_sym)
#     latest_trade_date_str = latest_trade_date.strftime('%d-%m-%Y - %H:%M:%S')
#     if ticker is not None:
#         data = dh.ReturnData(
#             ticker_sym,
#             start_date=latest_trade_date-relativedelta(years=1),
#             end_date = latest_trade_date,
#             limit=None
#             )
#     latest = data['Close'].iloc[-1]
#     diff = (data['Close'][-2]-data['Close'][-1])
#     pct_change = diff/data['Close'][-2]
#     pl_class = 'cgreen' if diff > 0 else 'cred'
#     pl_sign = '+' if diff > 0 else '-'
#     # pct_change = f''
#     ftwh = data['Close'].values.max()
#     ftwl = data['Close'].values.min()
#     #market_cap = data['market_cap'].iloc[-1]
#     # market_cap = 0
#     #curr = str(data['currency'].iloc[-1]).replace('EUR', '€')
#     curr = 'USD'
#     children = [
#         dbc.Row(
#             [
#                 html.H6(f'{ticker_sec} ({ticker_sym})', className='')
#             ]
#             ),
#         dbc.Row(
#             [
#                 dbc.Col([
#                     html.P(
#                     [
#                         html.Span(html.Strong(f'{curr} {latest:.2f} ', className=pl_class+' large bold')),
#                         html.Span(f'{pl_sign}{diff:.2f} ', className=pl_class+' small'),
#                         html.Span(f'({pl_sign}{pct_change:.2f}%)', className=pl_class+' small'),
#                         ],
#                     ),
#                     html.P(
#                         html.Span(f'As of {latest_trade_date_str}. Market {mo_str}.', className=' small')
#                         )
#                     ],
#                     md=12,
#                     ),
#                 ]),
#             ]
#     return children

@app.callback(
        Output('symbol_selected', 'value'),
    [
        Input('0stock_picker', 'value'),
    ]
    )
def select_symbol(symbol):
    return str(symbol)

@app.callback(
    Output('indicator_modal', 'is_open'),
    [
        Input('open_indicator_modal', 'n_clicks'),
        Input('close_indicator_modal', 'n_clicks')],
    [
        State('indicator_modal', 'is_open')
    ],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
        Output('selected_study_div', 'children'),
    [
        Input(f'set_{ind["value"]}', 'n_clicks') for ind in indicators
    ],
    [
        State('selected_study_div', 'children'),
        State('EMA10_win', 'value'),
    ],
)
def set_indicators(n_clicks, selected_studies, *args):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    selected_studies[button_id] = args
    return selected_studies

# def toggle_collapse(n, is_open):
#     if n:
#         return not is_open
#     return is_open
# for ind in indicators:
#     app.callback(
#         Output(f'{ind["value"]}_collapse', 'is_open'),
#         [Input(f'{ind["value"]}_collapse_button', 'n_clicks')],
#         [State(f'{ind["value"]}_collapse', 'is_open')],
#     )(toggle_collapse)


from dash.dependencies import ClientsideFunction

for ind in indicators:
    app.clientside_callback(
        ClientsideFunction(
            namespace='clientside',
            function_name='toggle_collapse'
        ),
        Output(f'{ind["value"]}_collapse', 'is_open'),
        [Input(f'{ind["value"]}_collapse_button', 'n_clicks')],
        [State(f'{ind["value"]}_collapse', 'is_open')]
    )
