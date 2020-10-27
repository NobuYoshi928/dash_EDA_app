# CSSの追加

import dash
import dash_core_components as dcc 
import dash_html_components as html 

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# 背景色と文字色を事前に設定しておく。
colors = {
    'background': 'blue',
    'text': '#7FDBFF'
}

app.layout = html.Div(
    # `style`を使用すればHTMLタグの`style`と同じようにCSS要素を適用できます。
    # 注意点としては`style`内では属性名をキャメルケースで指定します。
    # <div style="background-color: rgb(17, 17, 17)"></div>
    style={'backgroundColor': colors['background']}, 
    children=[
        html.H1(
            children='EDA自動化アプリ',
            # <h1 style="text-align: center; color: rgb(127, 219, 255);">Hello Dash</h1>
            style={'textAlign': 'center',
                   'color': colors['text']}),
        html.Div(
            children='csvファイルをアップロードして、基本的なEDA作業を自動化',
            # <div style="text-align: center; color: rgb(127, 219, 255);">
            style={'textAlign': 'center',
                   'color': colors['text']}),
        dcc.Graph(
            id='example-graph',
            figure={'data':[{'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                            {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'}],
                    # SVG要素内の属性`style`に適用される
                    'layout':{
                        'title': 'Dash Data Visualization',
                        'plot_bgcolor': colors['background'],
                        'paper_bgcolor': colors['background'],
                        'font': {'color': colors['text']}
                    }
            }
        )
])

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=5050, debug=True)