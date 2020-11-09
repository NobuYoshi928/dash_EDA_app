# ドロップダウン、マークダウン 、テキストボックス

# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

df = pd.read_csv('src/data/train.csv')
column_dropdown = [{'label':col, 'value':col} for col in df.columns]

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    # <label>Dropdown</label>
    html.Label('Dropdown'),
    dcc.Dropdown(
        options=[
            {'label': 'New York City', 'value': 'NYC'},
            {'label': u'Montréal', 'value': 'MTL'},
            {'label': 'San Francisco', 'value': 'SF'}
        ],
        # `value`で初期値を設定できる。
        value='MTL'
    ),
    # <label>Multi-Select Dropdown</label>
    html.Label('Multi-Select Dropdown'),
    dcc.Dropdown(
        options=[
            {'label': 'New York City', 'value': 'NYC'},
            {'label': u'Montréal', 'value': 'MTL'},
            {'label': 'San Francisco', 'value': 'SF'}
        ],
        value=['MTL', 'SF'],
        # `multi`で複数選択も可能にする。
        multi=True
    ),
    # <label>Radio Items</label>
    html.Label('Radio Items'),
    dcc.RadioItems(
        options=[
            {'label': 'New York City', 'value': 'NYC'},
            {'label': u'Montréal', 'value': 'MTL'},
            {'label': 'San Francisco', 'value': 'SF'}
        ],
        value='MTL'
    ),
    # <label>Checkboxes</label>
    html.Label('Checkboxes'),
    dcc.Checklist(
        options=[
            {'label': 'New York City', 'value': 'NYC'},
            {'label': u'Montréal', 'value': 'MTL'},
            {'label': 'San Francisco', 'value': 'SF'}
        ],
        value=['MTL', 'SF']
    ),
    # <label>Text Input</label>
    html.Label('Text Input'),
    # <input type="text" step="any" persisted_props="value" persistence_type="local" value="sample">
    dcc.Input(value='MTL', type='text'),

    # <label>Slider</label>
    html.Label('Slider'),
    dcc.Slider(
        min=0,
        max=9,
        marks={i: 'Label {}'.format(i) if i == 1 else str(i) for i in range(1, 6)},
        value=5,
    ),

# 基本的にDashでは属性の最後にCSSスタイルを指定します。
# <div style="column-count: 2;"><\div>
], style={'columnCount': 2})

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=5050, debug=True)