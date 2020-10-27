import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

from my_modules import utils

# global変数として定義しておく
df = pd.read_csv('src/data/train.csv')
target_column = None
use_df = None
cat_cols = None
num_cols = None

# pandasのデータフレームからHTMLタグを指定する。
def generate_table(dataframe, max_rows=None):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody(children=[
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
            ],
            style={
                'overflowX': 'scroll',
                'overflowY': 'scroll',
                'maxHeight': '500px'
            })
    ])

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.Div(children=[
        html.H4(children='データ概要'),
        generate_table(df, 5),
        html.H5('shape: {} raws × {} columns'.format(df.shape[0], df.shape[1]))
    ],
    style={'background': '#FFFFCC'}
    ),

    html.Div(children=[
        html.Div(children=[
            html.H5('目的変数を選択してください'),
            dcc.Dropdown(
                id='select-target',
                options=[{'label':column, 'value':column} for column in df.columns]
            ),
            html.Button(id="target-button", n_clicks=0, children="決定")
        ], style={'width': '40%'}
        ),

        html.Div(id='target_distribution', 
            children=html.H6('カラムが選択されていません'),
            style={
                'width': '40%',
                'background': '#CCFF99'
            }
        ),
        html.Br()
    ],
    style={'background': '#CCFF99'}
    ),

    html.Div(children=[
        html.H5('使用しないカラムを選択してください'),
        dcc.Dropdown(
            id='select-drop',
            options=[{'label':column, 'value':column} for column in df.columns],
            multi=True
        ),
        html.Button(id="drop-button", n_clicks=0, children="決定"),
        html.Div(id='use-column-list',
        children=html.H6('全てのカラムを使用します'))
    ],
    style={'background': '#99FFFF'}
    ),

    html.Div(children=[
        html.H5('数値変数の相関を調べる'),
        dcc.RadioItems(id='num_analysis_selection',
            options=[
                {'label': 'ペアプロット', 'value': 'AAA'},
                {'label': '相関行列', 'value': 'BBB'}
            ]
        ),
        html.Button(id="num_analysis-button", n_clicks=0, children="実行"),
        html.Div(id='num_result', 
            children=html.H5('分析方法を選択し、実行を押してください')
        )
    ])
])

# 目的変数の選択 -> 目的変数の分布を描画
@app.callback(
    Output(component_id='target_distribution', component_property='children'),
    [Input(component_id='target-button', component_property='n_clicks')],
    [State(component_id='select-target', component_property='value')]
)
def draw_target_distribution(n_clicks, input, *args):
    if input is None or input =='':
        return html.H5('カラムが選択されていません')
    else:
        global target_column
        target_column = input
        value_count_df = df[target_column].value_counts()
        fig = go.Figure(
            data=[go.Bar(x=value_count_df.index, y=value_count_df.values)],
            layout_title_text="目的変数（{}）の分布".format(target_column)
        )
        return dcc.Graph(figure=fig) 


# dropするカラムの選択 -> 確認 + 各global変数の上書き
@app.callback(
    Output(component_id='use-column-list', component_property='children'),
    [Input(component_id='drop-button', component_property='n_clicks')],
    [State(component_id='select-drop', component_property='value')]
)
def get_use_columns(n_clicks, drop_columns, *args):
    if target_column is None:
        return html.H5('先に目的変数を指定してください。')
    else:
        global use_df, cat_cols, num_cols
        use_df = df.copy()
        if drop_columns is None or drop_columns == []:
            return html.H5('全てのカラムを使用します')
        else:
            use_df.drop(drop_columns, axis=1, inplace=True)
            return html.H5('使用しないカラム：{}'.format(list(drop_columns)))
        cat_cols, num_cols = utils.get_cat_num_columns(use_df, target_column)


# 数値変数の分析方法の選択 -> 結果の描画
@app.callback(
    Output(component_id='num_result', component_property='children'),
    [Input(component_id='num_analysis-button', component_property='n_clicks')],
    [State(component_id='num_analysis_selection', component_property='value')]
)
def num_analysis(n_clicks, input, *args):
    if input is None or input =='':
        return html.H5('分析方法を選択し、実行を押してください')
    else:
        if input == 'AAA':
            fig = px.scatter_matrix(
                use_df, 
                dimensions=num_cols, 
                color=target_column
            )
            return dcc.Graph(figure=fig)
        else:
            return html.H5('分析方法を選択し、実行を押してください')




if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=5050, debug=True)