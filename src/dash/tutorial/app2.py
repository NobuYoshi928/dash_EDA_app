import base64
import io

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

from my_modules import utils

# 表示する数値の桁数を設定
pd.options.display.precision = 4

# global変数として定義しておく
df = None
target_column = None
use_df = None
cat_cols = 'BBB'
num_cols = 'AAA'

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

#--------------------------------CSS設定----------------------------------------
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# デフォルトのスタイルをアレンジ
common_style = {
    'font-family': 'Comic Sans MS', 
    'textAlign': 'center', 
    'margin': '0 auto'}
# アップロード部分のスタイル
upload_style={
    'width': '60%',
    'height': '60px',
    'lineHeight': '60px',
    'borderWidth': '1px',
    'borderStyle': 'dashed',
    'borderRadius': '5px',
    'textAlign': 'center',
    'margin': '0 auto'
}

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#--------------------------------レイアウト部分----------------------------------------
app.layout = html.Div(children=[
    html.Div([
        # ファイルアップロード
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'csvファイル、excelファイルをdrag and drop',
                html.A('(または選択)'),
                'してアップロード'
            ]),
            style=upload_style,
            multiple=True
        ),
        html.Br(),

        # アップロードしたデータの表示
        html.Div(id='output-data-upload_1')
    ]),

    html.Div(children=[
        # 目的変数の選択
        html.Div(
            children=[
                html.H5('目的変数を選択してください'),
                dcc.Dropdown(id='select-target'),
                html.Button(id="target-button", n_clicks=0, children="決定")
            ], 
            style={'width': '40%'}
        ),
        # 目的変数の分布の描画
        html.Div(
            id='target_distribution', 
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

    html.Div(
        children=[
            # 不要カラム選択
            html.H5('使用しない特徴量を選択してください'),
            dcc.Dropdown(
                id='select-drop',
                multi=True
            ),
            html.Button(id="drop-button", n_clicks=0, children="決定"),
            html.Div(
                id='use-column-list',
                children=html.H6('全てのカラムを使用します')
            )
        ],
        style={'background': '#99FFFF'}
    ),

    html.Div(
        children=[
            # 数値変数の分析方法選択
            html.H5('数値変数の相関を調べる'),
            dcc.RadioItems(
                id='num_analysis_selection',
                options=[
                    {'label': '統計量一覧', 'value': 'AAA'},
                    {'label': 'ペアプロット', 'value': 'BBB'},
                    {'label': '相関行列', 'value': 'CCC'}
                ]
            ),
            html.Button(id="num_analysis-button", n_clicks=0, children="実行"),
            # 数値変数の分析結果を描画
            html.Div(
                id='num_result', 
                children=html.H5('分析方法を選択し、実行を押してください')
            )
        ]
    )
])

# アップロードしたファイルをデータフレームとして読み込むための関数
def file_to_df(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    # csvデータをデータフレームに変換
    try:
        if 'csv' in filename:
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div(['このファイルは取り扱いできません。ファイル名を確認してください。'])
    return df

# DataFrameをdash_tableで表示するための変換用の関数
def df_processor(df):
    data_ = df.to_dict('records')
    columns_ = [{'name': i, 'id': i} for i in df.columns]
    return [data_, columns_]


# データアップロード -> 表の描画、global変数のtargetの上書き
@app.callback(Output('output-data-upload_1', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])
def update_output_1(list_of_contents, list_of_names):
    # ファイルがない時の自動コールバックを防ぐ
    if list_of_contents is None:
        raise dash.exceptions.PreventUpdate
    contents = [df_processor(file_to_df(c, n)) for c, n in zip(list_of_contents, list_of_names)]
    # 最新のファイルのデータのみ返す？
    return html.Div(children=[
                dash_table.DataTable(
                    column_selectable='multi',
                    data=contents[0][0],
                    columns=contents[0][1],
                    fixed_rows={'headers': True, 'data': 0},
                    style_table={
                        'overflowX': 'scroll',
                        'overflowY': 'scroll',
                        'maxHeight': '250px'
                    },
                    style_header={
                        'fontWeight': 'bold',
                        'textAlign': 'center'
                    }
                ),
                html.H5(children='shape: {} rows × {} columns'.format(df.shape[0], df.shape[1]))
            ])


# データアップロード -> target選択にカラムの情報を渡す
@app.callback(Output('select-target', 'options'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])
def setting_target_selection(list_of_contents, list_of_names):
    if list_of_contents is None:
        raise dash.exceptions.PreventUpdate
    global df
    df = [file_to_df(c, n) for c, n in zip(list_of_contents, list_of_names)][0]
    return [{'label':column, 'value':column} for column in df.columns]


# データアップロード -> drop選択にカラムの情報を渡す
@app.callback(Output('select-drop', 'options'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])
def setting_target_selection(list_of_contents, list_of_names):
    if list_of_contents is None:
        raise dash.exceptions.PreventUpdate
    global df
    df = [file_to_df(c, n) for c, n in zip(list_of_contents, list_of_names)][0]
    return [{'label':column, 'value':column} for column in df.columns]


# 目的変数の選択 -> 目的変数の分布を描画、global変数のtargetの上書き
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
        global use_df
        target_column = input
        use_df = df.copy()
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
    elif target_column in list(drop_columns):
        return html.H5('Error: {}は目的変数に指定されているため、選択できません。'.format(target_column))
    else:
        global use_df
        global cat_cols
        global num_cols
        use_df = df.copy()
        c, n = utils.get_cat_num_columns(use_df, target_column)
        cat_cols = c
        num_cols = n
        if drop_columns is None or drop_columns == []:
            return html.H5('全てのカラムを使用します')
        else:
            use_df.drop(drop_columns, axis=1, inplace=True)
            c, n = utils.get_cat_num_columns(use_df, target=None)
            cat_cols = c
            num_cols = n
            #return html.H5('使用しないカラム：{}'.format(target_column))
            return html.H5('使用しないカラム：{}'.format(list(drop_columns)))


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
            describe = use_df.describe().reset_index()
            return dash_table.DataTable(
                        column_selectable='multi',
                        fixed_rows={'headers': True, 'data': 0},
                        data=df_processor(describe)[0],
                        columns=df_processor(describe)[1],
                        style_table={
                            'overflowX': 'scroll',
                            'overflowY': 'scroll',
                            'maxHeight': '350px',
                            'maxWidht': '800px'
                        },
                        style_header={
                            'fontWeight': 'bold',
                            'textAlign': 'center'
                        }
                    )
        elif input == 'BBB':
            fig = px.scatter_matrix(
                use_df, 
                dimensions=num_cols, 
                color=target_column
            )
            return dcc.Graph(figure=fig)
        elif input == 'CCC':
            return html.H5(len(cat_cols))




if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=5050, debug=True)