# 表を描画する

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

# pandasでcsvデータを読み込む
df = pd.read_csv('src/data/train.csv')

# pandasのデータフレームからHTMLタグを指定する。
def generate_table(dataframe, max_rows=10):
    # 以下でHTMLの`table@タグを生成する
    # <table class></table>
    return html.Table([
        # テーブルヘッダーを指定する。
        # <thead class></thead>
        html.Thead(
            # `html.Tr` => <tr class></tr>
            # `html.Th` => <th class></th>
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        # テーブルの中身を指定する。
        # <tbody class></tbody>
        html.Tbody([
            # `html.Tr` => <tr class></tr>
            html.Tr([
                # `html.Td` => <td class></td>
                # 各行のデータの、各列のデータを順に出力
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            # 以下でDataFrameの中身を最大10行分生成する。
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H4(children='データ概要'),
    generate_table(df, 5)
])

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=5050, debug=True)