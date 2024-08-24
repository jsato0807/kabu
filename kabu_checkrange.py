import yfinance as yf
import pandas as pd

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
start_date = "2019-01-01"
end_date = "2024-01-01"

def calculate_range_period(df, threshold=0.02):
    # 最大値と最小値の差をレンジとして計算
    max_high = df['High'].max()
    min_low = df['Low'].min()
    range_width = max_high - min_low
    
    # 期間内でのレンジ幅がthresholdに基づいて評価
    longest_period = 0
    current_period = 0
    start_idx = 0

    for i in range(1, len(df)):
        if df['High'].iloc[i] - df['Low'].iloc[i] <= (max_high - min_low) * threshold:
            if current_period == 0:
                start_idx = i
            current_period += 1
        else:
            if current_period > longest_period:
                longest_period = current_period
            current_period = 0

    # 最後の期間のチェック
    if current_period > longest_period:
        longest_period = current_period

    return longest_period, range_width

def pareto_frontier(currency_pairs, start_date, end_date):
    pareto_set = []
    
    results = []

    # データを取得
    data = yf.download(currency_pairs, start=start_date, end=end_date)

    for pair in currency_pairs:
        df = data.xs(pair, level=1, axis=1)  # MultiIndexから該当通貨ペアのデータを取得

        # レンジ相場の期間と幅を計算
        longest_period, range_width = calculate_range_period(df)
        
        results.append((pair, longest_period, range_width))
    
    # Pareto Frontを計算
    for i, (pair_i, period_i, width_i) in enumerate(results):
        dominated = False
        for j, (pair_j, period_j, width_j) in enumerate(results):
            if i != j:
                # 比較して支配される場合
                if (period_j > period_i and width_j <= width_i) or (period_j >= period_i and width_j < width_i):
                    dominated = True
                    break
        if not dominated:
            pareto_set.append((pair_i, period_i, width_i))
    
    return pareto_set

# Pareto最適解の通貨ペアを探す
pareto_optimal_pairs = pareto_frontier(currency_pairs, start_date, end_date)
print("Pareto Optimal Pairs:")
for pair, period, width in pareto_optimal_pairs:
    print(f"{pair}: Longest Period = {period}, Range Width = {width}")
