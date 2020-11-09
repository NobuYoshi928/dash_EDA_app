# モジュールのインポート
import base64
import io

import dash
import dash_core_components as dcc
import dash_html_components as html

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
