import base64
import io

import category_encoders as ce
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler

from my_modules import utils


# global変数として定義しておく
df = None
target_column = None
use_df = None
processed_df = None
cat_cols = []
num_cols = []

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
    'background':'white',
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
        html.H1('分類タスク ベースモデル生成アプリ'),
        html.H5('・テーブルデータの二値分類タスクについて、ロジスティック回帰によるベースラインモデルを作成することができます。'),
        html.H5('・欠損値がある場合は、Step4で欠損値補完処理を行ってください。'),
        html.H5('・文字列データを含む場合は、Step4でOne-Hotエンコーディング処理を行ってください（欠損値はそのまま処理することができます）。'),
        html.H5('・操作を途中からやり直す場合、それ以降の処理はリセットされます。再度各処理の実行ボタンを押してください。')
        ]
    ),
        
    html.Div(children=[
        html.H3('Step1: データのアップロード'),

        # ファイルアップロード
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'csvファイル、xlsファイルをdrag and drop',
                html.A('(または選択)'),
                'してアップロード'
            ]),
            style=upload_style,
            multiple=True
        ),
        html.Br(),

        # アップロードしたデータの表示
        html.Div(id='output-data-upload_1')
        ],
        style={
            'background': '#EAD9FF',
            'height':'450px',
            'width':'1400px'}
    ),

    html.Div([
        html.Div(children=[
            html.H3('Step2: 目的変数を設定（必須）'),
            # 目的変数の選択
            html.Div(
                children=[
                    html.H5('目的変数を選択してください'),
                    dcc.Dropdown(
                        id='select-target',
                        style={
                            'width':'80%',
                            'float':'left'
                        }
                    ),
                    html.Button(id="target-button", n_clicks=0, children="決定", style={'background': '#DDDDDD'})
                ]
            ),
            # 目的変数の分布の描画
            html.Div(
                id='target_distribution'
            )
        ],
        style={
            'background': '#AEFFBD',
            'float':'left',
            'width':'600px',
            'height':'700px'
            }
        ),

        html.Div([
            html.Div(
                children=[
                    html.H3('Step3: 特徴量選択（任意）'),
                    # 不要カラム選択
                    html.H5('使用しない特徴量を選択してください'),
                    dcc.Dropdown(
                        id='select-drop',
                        multi=True,
                        style={'width': '80%', 'float':'left'}
                    ),
                    html.Button(id="drop-button", n_clicks=0, children="決定", style={'background': '#DDDDDD'}),
                    html.Div(
                        id='use-column-list',
                        children=html.H6('全てのカラムを使用します')
                    )
                ],
                style={
                    'background': '#CCFFFF',
                    'height':'350px',
                    'width':'800px',
                    'float':'left'
                }
            ),

            html.Div(
                children=[
                    html.H3('Step4: 前処理を選ぶ（任意）'),
                    # 前処理の選択
                    html.H5('前処理方法を選択してください'),
                    dcc.Checklist(
                        id='select-preprocess',
                        options=[
                            {'label': '欠損値補完（数値変数を平均値で補完する）', 'value': 'FN'},
                            {'label': '数値変数のYeo-Johnson変換', 'value': 'YJ'},
                            {'label': '数値変数の標準化', 'value': 'SS'},
                            {'label': 'カテゴリ変数のOne-Hot Encodeing', 'value': 'OE'}
                        ]
                    ),
                    html.Button(id="preprocess-button", n_clicks=0, children="決定", style={'background': '#DDDDDD'}),
                    html.Div(
                        id='preprocess_result',
                        children=html.H6('前処理は行われていません')
                    )
                ],
                style={
                    'background': '#FFFFDD',
                    'height':'350px',
                    'width':'800px',
                    'float':'left'
                }
            )
        ],
        style={
            'height':'700px',
            'width':'700px',
            'float':'left'
        }
        )
    ],
    style={
        'float':'left',
        'height':'700px',
        'width':'1400px'
    }
    ),

    html.Div(
        children=[
            html.H3('Step5: 数値変数の分析'),
            # 数値変数の分析方法選択
            html.H5('分析方法を選択し、実行を押してください'),
            dcc.RadioItems(
                id='num_analysis_selection',
                options=[
                    {'label': '統計量一覧', 'value': 'AAA'},
                    {'label': 'ペアプロット', 'value': 'BBB'},
                    {'label': '相関行列', 'value': 'CCC'}
                ]
            ),
            html.Button(id="num_analysis-button", n_clicks=0, children="実行", style={'background': '#DDDDDD'}),
            # 数値変数の分析結果を描画
            html.Div(
                id='num_result', 
                style={'textAlign':'center'}
            )
        ],
        style={
            'background': '#FFE4B5',
            'float':'left',
            'width':'1400px'
        }
    ),

    html.Div(
        children=[
            html.H3('Step6: カテゴリ変数の分析'),
            # カテゴリ変数の分析方法選択
            html.H5('目的変数との連関係数を調べる'),
            html.Button(id="cat_analysis-button", n_clicks=0, children="実行", style={'background': '#DDDDDD'}),
            # カテゴリ変数の分析結果を描画
            html.Div(
                id='cat_result',
                children=''
            )
        ],
        style={
            'background': '#FFD5EC',
            'float':'left',
            'width':'1400px'
        }
    ),

    html.Div(
        children=[
            html.H3('Step7: モデル学習（ロジスティック回帰）'),
            html.Div(children=[
                html.H6(id='parameta_C'),
                dcc.Slider(
                    id='rogistic_parameta', 
                    min=-4,
                    max=2,
                    step=0.01,
                    value=0,
                    marks={
                        -4: {'label': '0.0001'},
                        -3: {'label': '0.001'},
                        -2: {'label': '0.01'},
                        -1: {'label': '0.1'},
                        0: {'label': '1'},
                        1: {'label': '10'},
                        2: {'label': '100'}
                    }
                )],
                style={'width':'40%'}
            ),
            html.Button(id="train-button", n_clicks=0, children="学習を開始する", style={'background': '#DDDDDD'}),
            # カテゴリ変数の分析結果を描画
            html.Div(
                id='train_result',
                children=''
            )
        ],
        style={
            'background': '#EEEEEE',
            'width':'1400px'}
    )
])

#--------------------------------Callback部分----------------------------------------

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


# データアップロード -> 表の描画
@app.callback(Output('output-data-upload_1', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])
def update_output_1(list_of_contents, list_of_names):
    # ファイルがない時の自動コールバックを防ぐ
    if list_of_contents is None:
        raise dash.exceptions.PreventUpdate
    
    global df
    df = [file_to_df(c, n) for c, n in zip(list_of_contents, list_of_names)][0]
    contents = [df_processor(file_to_df(c, n)) for c, n in zip(list_of_contents, list_of_names)]
    # ファイルをアップロードし直した時のために他のグローバル変数もリセット
    global target_column
    global use_df
    global processed_df
    global cat_cols
    global num_cols
    target_column = None
    use_df = None
    processed_df = None
    cat_cols = []
    num_cols = []
    # 最新のファイルのデータのみ返す？
    return dcc.Loading(
        html.Div(children=[
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
    )


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


# 目的変数の選択 -> 目的変数の分布を描画
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
        global processed_df
        global cat_cols
        global num_cols
        target_column = input
        use_df = df.copy()
        processed_df = df.copy()
        c, n = utils.get_cat_num_columns(use_df, target_column)
        cat_cols = c
        num_cols = n
        value_count_df = df[target_column].value_counts()
        fig = go.Figure(
            data=[go.Bar(x=value_count_df.index, y=value_count_df.values)],
            layout_title_text="目的変数（{}）の分布".format(target_column)
        )
        return [dcc.Graph(
                    figure=fig, 
                    style={
                        'background': '#AEFFBD', 
                        'textAlign': 'center',
                        'padding':'20px'
                    }
                ),
                html.Br()] 


# dropするカラムの選択 -> 確認文章の表示
@app.callback(
    Output(component_id='use-column-list', component_property='children'),
    [Input(component_id='drop-button', component_property='n_clicks')],
    [State(component_id='select-drop', component_property='value')]
)
def get_use_columns(n_clicks, drop_columns, *args):
    global use_df
    global processed_df
    global cat_cols
    global num_cols
    if n_clicks == 0:
        return html.H5('')
    elif target_column is None:
        return html.H5('先に目的変数を指定してください。')
    elif drop_columns is None or drop_columns == []:
        use_df = df.copy()
        processed_df = use_df.copy()
        c, n = utils.get_cat_num_columns(use_df, target_column)
        cat_cols = c
        num_cols = n
        return html.H5('全てのカラムを使用します')
    elif target_column in list(drop_columns):
        return html.H5('Error: {}は目的変数に指定されているため、選択できません。'.format(target_column))
    else:
        use_df = df.copy()
        use_df.drop(drop_columns, axis=1, inplace=True)
        processed_df = use_df.copy()
        c, n = utils.get_cat_num_columns(use_df, target_column)
        cat_cols = c
        num_cols = n
        return html.H5('使用しないカラム：{}'.format(list(drop_columns)))


# 前処理の選択 -> 行った前処理の表示
@app.callback(
    Output(component_id='preprocess_result', component_property='children'),
    [Input(component_id='preprocess-button', component_property='n_clicks')],
    [State(component_id='select-preprocess', component_property='value')]
)
def preprocessing(n_clicks, preprocessings):
    global use_df
    global processed_df
    global cat_cols
    global num_cols
    if n_clicks == 0:
        return html.H5('')
    elif target_column is None:
        return html.H5('先に目的変数を指定してください。')
    elif preprocessings is None or preprocessings == []:
        processed_df = use_df.copy()
        return html.H5('前処理は行われていません')
    else:
        text = []
        processed_df = use_df.copy()
        if 'FN' in str(preprocessings):
            processed_df = processed_df.fillna(processed_df.mean())
            text.append('欠損値補完')
        if 'YJ' in str(preprocessings):
            yj = PowerTransformer(method='yeo-johnson')
            processed_df[num_cols] = yj.fit_transform(processed_df[num_cols])
            text.append('Yeo-Johnson変換')
        if 'SS' in str(preprocessings):
            ss = StandardScaler()
            processed_df[num_cols] = ss.fit_transform(processed_df[num_cols])
            text.append('標準化')
        if 'OE' in str(preprocessings):
            oe = ce.OneHotEncoder(cols=cat_cols,handle_unknown='impute')
            processed_df = oe.fit_transform(processed_df)
            text.append('One-Hot Encoding')
        return html.H5('{}を行いました。'.format(text))


# 数値変数の分析方法の選択 -> 結果の描画
@app.callback(
    Output(component_id='num_result', component_property='children'),
    [Input(component_id='num_analysis-button', component_property='n_clicks')],
    [State(component_id='num_analysis_selection', component_property='value')]
)
def num_analysis(n_clicks, input, *args):
    if input is None or input =='':
        return html.H5('')
    elif use_df is None:
        return html.H5('目的変数が選択されていません。')
    else:
        # 統計量一覧表の描画
        if input == 'AAA':
            describe = processed_df[num_cols].describe().round(4).reset_index()
            return [dash_table.DataTable(
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
                    ),
                    html.Br()]
        # ペアプロットの描画
        elif input == 'BBB':
            fig = px.scatter_matrix(
                processed_df, 
                dimensions=num_cols, 
                color=target_column
            )
            fig.update_layout(
                dragmode='select',
                width=1000,
                height=600,
                hovermode='closest'
            )
            return [dcc.Graph(figure=fig, style={'textAlign': 'center', 'padding':'20px'}),
                    html.Br()]
        # 相関係数（ヒートマップ）の描画
        elif input == 'CCC':
            corr = processed_df[num_cols].corr().round(4)
            fig = ff.create_annotated_heatmap(
                z=corr.values, 
                x=list(corr.columns),
                y=list(corr.index), 
                colorscale='Oranges',
                hoverinfo='none'
            )
            return [dcc.Graph(figure=fig, style={'textAlign': 'center', 'padding':'20px'}),
                    html.Br()]


# カテゴリ変数の連関係数を求める
@app.callback(
    Output(component_id='cat_result', component_property='children'),
    [Input(component_id='cat_analysis-button', component_property='n_clicks')]
)
def cat_analysis(n_clicks, *args):
    if n_clicks == 0:
        return html.H5('')
    else:
        if input is None or input =='':
            return html.H5('分析方法を選択し、実行を押してください')
        elif use_df is None:
            return html.H5('目的変数が選択されていません。')
        else:
            global cat_cols
            child = []
            for col in cat_cols:
                child.append(html.H5('{}との連関係数： {}'.format(col, utils.cramersV(use_df[col], use_df[target_column]))))
            return child


# 正則化パラメータの値を反映させる
@app.callback(
    Output(component_id='parameta_C', component_property='children'),
    [Input(component_id='rogistic_parameta', component_property='value')]
)
def parameta_C(C, *args):
    return '正則化係数：{}'.format(10**C)


# ロジスティック回帰
@app.callback(
    Output(component_id='train_result', component_property='children'),
    [Input(component_id='train-button', component_property='n_clicks')],
    [State(component_id='rogistic_parameta', component_property='value')]
)
def model_training(n_clicks, C, *args):
    if n_clicks==0:
        return html.H5('開始ボタンを押すと、学習が始まります')
    else:
        # データの分割
        X = processed_df.drop(target_column, axis=1)
        y = processed_df.loc[:, target_column]
        if y.dtype == 'object':
            y = y.apply(lambda x: 1 if x == list(y.value_counts().index)[0] else 0)
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size = 0.2, random_state=928)
        # 学習+推論
        model = LogisticRegression(C = 10**C, class_weight='balanced')
        model.fit(X_tr, y_tr)
        y_pred_tr = model.predict_proba(X_tr)[:,1]
        y_pred_va = model.predict_proba(X_va)[:,1]
        # ROC曲線の描画
        fig1 = go.Figure()
        fig1.add_shape(type='line', line=dict(dash='dash'),x0=0, x1=1, y0=0, y1=1)
        fpr_tr, tpr_tr, _ = roc_curve(y_tr, y_pred_tr)
        fpr_va, tpr_va, _ = roc_curve(y_va, y_pred_va)
        fig1.add_trace(go.Scatter(x=fpr_tr, y=tpr_tr, name="train", mode='lines'))
        fig1.add_trace(go.Scatter(x=fpr_va, y=tpr_va, name="valid", mode='lines'))
        fig1.update_layout(
            title='ROC曲線',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=500, height=500
        )
        # 回帰係数の描画
        fig2 = go.Figure(
            data=go.Bar(x=X_tr.columns, y=model.coef_.flatten()),
            layout = go.Layout(
                title='変数重要度（回帰係数）',
                xaxis=dict(title='カラム名')
            )
        )
        # AUCスコアと合わせて出力
        return [html.Div([
                    dcc.Graph(figure=fig1, style={'float':'left', 'padding':'20px', 'height':'500px'}),
                    dcc.Graph(figure=fig2, style={'float':'left', 'padding':'20px', 'height':'500px'})
                ]),
                html.H5('AUC(train):{}'.format(roc_auc_score(y_true=y_tr, y_score=y_pred_tr))),
                html.H5('AUC(valid):{}'.format(roc_auc_score(y_true=y_va, y_score=y_pred_va))),
                html.Br()]


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=5050, debug=True)