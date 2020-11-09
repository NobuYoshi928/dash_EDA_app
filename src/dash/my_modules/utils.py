import dash
import dash_html_components as html
import dash_core_components as dcc
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# pandasのデータフレームからHTMLタグを指定する。
def generate_table(dataframe, max_rows=10):
    # 以下でHTMLの`table@タグを生成する
    # <table class></table>
    return html.Table([
        # テーブルヘッダーを指定
        html.Thead(html.Tr([html.Th(col) for col in dataframe.columns])),
        # テーブルの中身を指定する。
        html.Tbody([
            # `html.Tr` => <tr class></tr>
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            # 以下でDataFrameの中身を最大10行分生成する。
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

def get_cat_num_columns(df, target):
    '''
    入力：df(DataFrame)
    出力１：カテゴリ変数のカラム名のリスト
    出力２：数値変数のカラム名のリスト」
    '''
    # 目的変数が指定されたときは予め除く
    feature_cols = list(df.columns)
    if target:
        feature_cols.remove(target)
    
    # カテゴリ変数/数値変数のカラム名が入る空のリストを作成
    cat_col_list = [column for column in feature_cols if df[column].dtype == 'object']
    num_col_list = [column for column in feature_cols if df[column].dtype != 'object']
    
    return cat_col_list, num_col_list


def null_rate_check(df, top=None):
    '''
    入力：df(DataFrame), top(int)上位何件表示するか
    出力：欠損率が高いカラム順に表示
    '''
    return (df.isnull().sum()/len(df)).sort_values(ascending=False)[:top]

# クラメールの連関係数
def cramersV(x, y):
    table = np.array(pd.crosstab(x, y)).astype(np.float32)
    n = table.sum()
    colsum = table.sum(axis=0)
    rowsum = table.sum(axis=1)
    expect = np.outer(rowsum, colsum) / n
    chisq = np.sum((table - expect) ** 2 / expect)
    return np.sqrt(chisq / (n * (np.min(table.shape) - 1)))

    
def agg_features_generator(df, aggregation_settings):
    '''
    概要：データフレームから、集約特徴量を作成する関数
    入力1 df：(DataFrame)
    入力2 agg_setting：(list)リスト形式で集約する特徴量を選択する（複数可）
         ※集約法は'mean','median', 'max', 'min', 'std'など 
            [
                ([カテゴリ変数カラム名, ...], [(数値変数カラム名, 集約法), (数値変数カラム名, 集約法), ...]),
                ([カテゴリ変数カラム名, ...], [(数値変数カラム名, 集約法), (数値変数カラム名, 集約法), ...]),
                ...
            ]
    出力：集約特徴量が追加されたDataFrame
    '''
    groupby_aggregate_names = []
    for groupby_cols, specs in aggregation_settings:
        group_object = df.groupby(groupby_cols)
        for select, agg in specs:
            groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
            df = df.merge(group_object[select]
                                 .agg(agg)
                                 .reset_index()
                                 .rename(index=str,
                                         columns={select: groupby_aggregate_name})
                                 [groupby_cols + [groupby_aggregate_name]],
                                 on=groupby_cols,
                                 how='left')
            groupby_aggregate_names.append(groupby_aggregate_name)