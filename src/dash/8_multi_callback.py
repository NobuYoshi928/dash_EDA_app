import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

app = dash.Dash()

app.layout = html.Div([
    dcc.Dropdown(
        id="dropdown-a",
        options=[{"label": i, "value": i} for i in ["Canada", "USA", "Mexico"]],
        value="Canada",
    ),
    html.Div(id="output-a"),

    dcc.RadioItems(
        id="dropdown-b",
        options=[{"label": i, "value": i} for i in ["MTL", "NYC", "SF"]],
        value="MTL"
    ),
    html.Div(id="output-b")
])

# 1つ目の出力
@app.callback(
    Output("output-a", "children"),
    [Input("dropdown-a", "value")]
)
def callback_a(dropdown_value):
    return "You have selected {}".format(dropdown_value)

# 2つ目の出力
@app.callback(
    Output("output-b", "children"),
    [Input("dropdown-b", "value")]
)
def callback_b(dropdown_input):
    return "You have selected {}".format(dropdown_input)

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=5050, debug=True)