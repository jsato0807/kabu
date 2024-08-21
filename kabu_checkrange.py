import yfinance as yf
import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real

# 通貨ペアのリスト
currency_pairs = [
    "USDJPY=X", "EURUSD=X", "GBPUSD=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X",
    "EURJPY=X", "EURGBP=X", "EURAUD=X", "EURCAD=X", "EURCHF=X", "EURNZD=X",
    "GBPJPY=X", "GBPAUD=X", "GBPCAD=X", "GBPCHF=X", "GBPNZD=X",
    "AUDJPY=X", "AUDCAD=X", "AUDCHF=X", "AUDNZD=X",
    "CADJPY=X", "CADCHF=X", "CHFJPY=X",
    "NZDJPY=X", "NZDCHF=X"
]

# データを取得する期間
start_date = "2010-01-01"
end_date = "2024-01-01"

def download_data(currency_pairs, start_date, end_date):
    # 通貨ペアのリストからデータを一度に取得
    data = yf.download(currency_pairs, start=start_date, end=end_date)
    return data

def calculate_range_and_period(df, window):
    # 一定期間ごとの最高値と最安値の差（レンジ）を計算
    rolling_max = df['High'].rolling(window=window).max()
    rolling_min = df['Low'].rolling(window=window).min()
    range_ = rolling_max - rolling_min
    
    # レンジの平均値
    mean_range = range_.mean()
    
    # 期間のうち、レンジが平均値以下の割合を計算
    percentage_in_range = (range_ < mean_range).mean()
    
    return mean_range, percentage_in_range

def objective_function(params, df):
    range_weight, period_weight = params
    mean_range, percentage_in_range = calculate_range_and_period(df,len(df))
    
    # レンジが狭いほど、期間が長いほどスコアが高くなるようにスコアを計算
    range_score = 1 / mean_range  # レンジが狭いほどスコアが高い
    period_score = percentage_in_range  # 期間が長いほどスコアが高い
    
    # range_weight と period_weight による重み付きスコアの合成
    score = range_weight * range_score + period_weight * period_score
    
    return -score  # gp_minimizeは最小化を行うため、負のスコアを返す

def find_best_range_pairs(currency_pairs, start_date, end_date, n_best=5):
    # データを一度に取得
    data = download_data(currency_pairs, start_date, end_date)
    
    results = []
    
    for pair in currency_pairs:
        # 各通貨ペアのデータを取り出す
        df = data['Adj Close'].loc[:, pair].to_frame()
        df.columns = ['Adj Close']  # Adjust column name to match other columns
        
        # パラメータ空間の定義
        space = [Real(0.1, 10.0, name='range_weight'), Real(0.1, 10.0, name='period_weight')]
        
        # ベイズ最適化の実行
        res = gp_minimize(lambda params: objective_function(params, df), space, n_calls=50, random_state=0)
        
        # 最適なスコアとパラメータを記録
        for i, (params, score) in enumerate(zip(res.x_iters, res.func_vals)):
            results.append((pair, params, -score))  # スコアは負にして記録
    
    # スコアの高い順にソート
    results.sort(key=lambda x: x[2], reverse=True)
    
    # ベストNを取得
    best_results = results[:n_best]
    
    return best_results

best_pairs = find_best_range_pairs(currency_pairs, start_date, end_date, n_best=5)
for pair, params, score in best_pairs:
    print(f"Currency Pair: {pair}, Parameters: {params}, Score: {score}")
