# Markdownを使う

import dash
import dash_core_components as dcc
import dash_html_components as html


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# 出力させたいMarkdownでの文章を変数に格納しておく。
markdown_text = '''
# 11111
## 22222
### 33333
#### 44444

- aaaaaaaaaaa
- bbbbbbbbbbb
- ccccccccccc
'''

app.layout = html.Div([
    dcc.Markdown(children=markdown_text)
])

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=5050, debug=True)