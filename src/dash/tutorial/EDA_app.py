import base64
import io

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table

import pandas as pd
import numpy as np
import plotly.graph_objs as go

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

# データフレームから表を生成する関数
def generate_table(dataframe, max_rows=10):
    return html.Table([
        # テーブルヘッダーを指定する。
        html.Thead(html.Tr([html.Th(col) for col in dataframe.columns])),
        # テーブルの中身を指定する。
        html.Tbody([
            html.Tr([
                # 各行のデータの、各列のデータを順に出力
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            # 以下でDataFrameの中身を最大10行分生成する。
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

# アプリの実態(インスタンス)を定義
app = dash.Dash(__name__)

# アプリの見た目を記述
app.layout = html.Div(
    html.Div([
        html.H1('EDA自動化アプリ'),
        # 空白を加える
        html.Br(),

        # ファイルアップロード部分
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'csvファイル、excelファイルをdrag and drop',
                html.A('(または選択)'),
                'してアップロード'
            ]),
            style=upload_style,
            # Allow multiple files to be uploaded
            # # これないとおそらくエラーになる
            multiple=True
        ),
        html.Br(),

        # アップロードしたファイルをデータテーブルとして表示させる部分
        html.Div(
            children=[
                dash_table.DataTable(
                    id='output-data-upload_1',
                    column_selectable='multi',
                    fixed_rows={'headers': True, 'data': 0},
                    style_table={
                        'overflowX': 'scroll',
                        'overflowY': 'scroll',
                        'maxHeight': '250px'
                    },
                    style_header={
                        'fontWeight': 'bold',
                        'textAlign': 'center'}
                )
            ],
            style={
                'height': '300px'
        }),

        # .info()の結果を表示させる部分
        html.Div(
            id='output-data-upload_2',
            style={
                'height': '300px'
            }
        )
    ]),
    style=common_style
)

# アップロードしたファイルをデータフレームとして読み込むための関数
def file_to_df(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    # step1:csvデータをデータフレームに変換
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

# データフレームを扱いやすいように変形
def df_processor(df):
    data_ = df.to_dict('records')  #1レコードずつのリスト型に [{A:0, B:2,...}, {A:3, B:8,...}, ...]（引数により変えられる）
    columns_ = [{'name': i, 'id': i} for i in df.columns] #[{'name':A, 'id':A}, {'name':B, 'id':B}, ...]
    return [data_, columns_]

# pandasのデータフレームからHTMLタグを指定する。
def generate_table(dataframe, max_rows=30):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


#
@app.callback([Output('output-data-upload_1', 'data'),
               Output('output-data-upload_1', 'columns')],
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])
def update_output_1(list_of_contents, list_of_names):
    # ファイルがない時の自動コールバックを防ぐ
    if list_of_contents is None:
        raise dash.exceptions.PreventUpdate
    contents = [df_processor(file_to_df(c, n)) for c, n in zip(list_of_contents, list_of_names)]
    # 最新のファイルのデータのみ返す？
    return [contents[0][0], contents[0][1]]


@app.callback([Output('output-data-upload_2', 'children')],
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])
def update_output_2(list_of_contents, list_of_names):
    # ファイルがない時の自動コールバックを防ぐ
    if list_of_contents is None:
        raise dash.exceptions.PreventUpdate
    df_list = [file_to_df(c, n) for c, n in zip(list_of_contents, list_of_names)]
    df = df_list[0]
    return generate_table(df)



if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=5050, debug=True)