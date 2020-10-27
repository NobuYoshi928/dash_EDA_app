# やや複雑なグラフを描画する

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv('https://gist.githubusercontent.com/chriddyp/5d1ea79569ed194d432e56108a04d188/raw/a9f9e8076b837d541398e999dcbac2b2826a81f8/gdp-life-exp-2007.csv')


app.layout = html.Div([
    # `Graph`は`plotly.js`を使ってレンダリングを行う。
    # SVGとWebGLを使用しており35種類以上のグラフに対応している。
    dcc.Graph(
        id='life-exp-vs-gdp',
        # この`figure`属性は`plotly.py`と同じ挙動を示す。
        # https://plotly.com/python
        figure={
            # plotlyでは出力したいデータをリストで指定する。
            # 複数のデータを使用する場合には辞書のリストを指定する。
            'data': [
                # 大陸毎のデータを1かたまりにして出力する。
                dict(
                    x=df[df['continent'] == i]['gdp per capita'],
                    y=df[df['continent'] == i]['life expectancy'],
                    text=df[df['continent'] == i]['country'],
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=i
                ) for i in df.continent.unique()
            ],
            'layout': dict(
                xaxis={'type': 'log', 'title': 'GDP Per Capita'},
                yaxis={'title': 'Life Expectancy'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                # カーソルを当てた際に最も近いデータ点の情報を出力する。
                hovermode='closest'
            )
        }
    )
])

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=5050, debug=True)