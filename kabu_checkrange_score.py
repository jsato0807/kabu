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
start_date = "2019-05-01"
end_date = "2024-01-01"

def calculate_range_period(df, threshold=0.02):
    df = df.copy()  # 明示的にコピーを作成
    df.loc[:, 'mid'] = (df['High'] + df['Low']) / 2
    df.loc[:, 'range'] = df['High'] - df['Low']
    
    longest_period = 0
    best_range_width = None
    
    current_period = 0
    start_idx = 0
    
    for i in range(1, len(df)):
        if df['range'].iloc[i] <= df['mid'].iloc[i] * threshold:
            if current_period == 0:
                start_idx = i
            current_period += 1
        else:
            if current_period > longest_period:
                longest_period = current_period
                best_range_width = df['range'].iloc[start_idx:i].max()
            current_period = 0
    
    return longest_period, best_range_width

def calculate_score(period, width, max_period, min_period, max_width, min_width, alpha=0.5, beta=0.5):
    norm_period = (period - min_period) / (max_period - min_period)
    norm_width = (width - min_width) / (max_width - min_width)
    return alpha * norm_period + beta * norm_width

def pareto_frontier_with_scores(currency_pairs, start_date, end_date):
    pareto_set = []
    
    results = []

    # データを取得
    data = yf.download(currency_pairs, start=start_date, end=end_date)

    periods = []
    widths = []

    # 各通貨ペアの評価
    for pair in currency_pairs:
        df = data.xs(pair, level=1, axis=1)
        longest_period, range_width = calculate_range_period(df)
        
        periods.append(longest_period)
        widths.append(range_width)
        results.append((pair, longest_period, range_width))
    
    max_period = max(periods)
    min_period = min(periods)
    max_width = max(widths)
    min_width = min(widths)

    # スコア計算
    scored_results = [(pair, period, width, calculate_score(period, width, max_period, min_period, max_width, min_width)) for pair, period, width in results]
    
    # スコアでソート
    scored_results.sort(key=lambda x: x[3], reverse=True)

    # パレート最適解を選定
    for i, (pair_i, period_i, width_i, score_i) in enumerate(scored_results):
        dominated = False
        for j, (pair_j, period_j, width_j, score_j) in enumerate(scored_results):
            if i != j:
                # スコアで支配される場合
                if (score_j > score_i) and (period_j >= period_i and width_j <= width_i) or (score_j > score_i) and (period_j > period_i and width_j < width_i):
                    dominated = True
                    break
        if not dominated:
            pareto_set.append((pair_i, period_i, width_i))
    
    return pareto_set

# Pareto最適解の通貨ペアを探す
pareto_optimal_pairs_scores = pareto_frontier_with_scores(currency_pairs, start_date, end_date)

print("Pareto Optimal Pairs (With Scores):")
for pair, period, width in pareto_optimal_pairs_scores:
    print(f"{pair}: Longest Period = {period}, Range Width = {width}")
