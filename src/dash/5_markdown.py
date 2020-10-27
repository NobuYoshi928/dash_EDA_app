# Markdownを使う

import dash
import dash_core_components as dcc
import dash_html_components as html


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# 出力させたいMarkdownでの文章を変数に格納しておく。
markdown_text = '''
### EDA自動化アプリ

作った動機
- 成果物を作りたい
- 自分で使えるものが良い
- EDAって最初ほとんど同じ
'''

app.layout = html.Div([
    dcc.Markdown(children=markdown_text)
])

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=5050, debug=True)