# -*- coding: utf-8 -*-

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

import pandas as pd

from app import app
#from app import server
from apps import app1, app2

from atomm.DataManager.main import MSDataManager
dh = MSDataManager()
options = dh.IndexConstituentsDict('SPY')
spy_info = pd.read_csv('./data/spy.csv')

server = app.server

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
            dbc.Input(
                id='0stock_picker',
                list='list-suggested-inputs',
                placeholder='Search for stock',
                value='MMM - 3M Company',
                type='text',
                size='50',
                className='bg-dark border-light text-light w-100 rounded br-3 pl-2 pr-2 pt-1 pb-1'
                ),
            className='border-light'
            ),
    ],
    no_gutters=True,
    className="ml-auto flex-nowrap mt-3 mt-md-0",
    align="center",
)

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
            dbc.NavLink(
                'Overview',
                href='/apps/app1',
                id='link_app1',
                ),
            dbc.NavLink('Machine Learning', href='/apps/app2', id='link_app2'),
            dbc.Collapse(
                search_bar,
                id="navbar-collapse",
                navbar=True,
                className='mx-auto display-block'
                ),  
        ],
        color="dark",
        dark=True,
        className='navbar-dark ml-auto mr-auto shadow',
        sticky='top'
    )
    return navbar


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(create_navbar()),
    html.Div(id='page-content')
])


@app.callback(
    Output('page-content', 'children'),
    [
     Input('url', 'pathname')
     ]
    )
def display_page(pathname):
    if pathname == ('/apps/app1' or '/'):
        return app1.layout
    if pathname == '/apps/app2':
        return app2.layout
    else:
        return '404'


if __name__ == '__main__':
    app.run_server(debug=False)
