# 必要なライブラリをインポート。
import dash
import dash_core_components as dcc  #描画に必要なUIを提供するパッケージ
import dash_html_components as html #HTMLタグを提供するパッケージ

# カスタムCSSのパスをリスト形式で指定
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# ファイル名をアプリ名として起動。その際に外部CSSを指定できる。
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# `layout`にアプリの外観部分を指定していく
app.layout = html.Div(children=[

    html.H1(children='EDA自動化アプリ'),
    html.Div(children='csvファイルをアップロードして、基本的なEDA作業を自動化'),
    dcc.Graph(
        id='example-graph',
        figure={
            'data':[{'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                    {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'}],
            'layout':{'title': 'Dash Data Visualization'}
        }
    )
])

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=5050, debug=True)


"""
<div id="_dash-app-content" class>
    <div>
        <h1>Hello Dash</h1>
        <div>Dash: A web application framework for Python.</div>
        <div id="exampl-graph" class="dash-graph">
            <!-- 描画内容 -->
        </div>
    </div>
</div>
"""