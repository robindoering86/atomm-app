#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 22:32:21 2020

@author: robin
"""

# -*- coding: utf-8 -*-

import dash
import dash_bootstrap_components as dbc


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], url_base_pathname='/')
#server = app.server
app.config.suppress_callback_exceptions = True
app.title = 'atomm web app'
