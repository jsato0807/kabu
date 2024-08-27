import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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

def calculate_range_period(df, threshold=0.1):
    max_high = df['High'].max()
    min_low = df['Low'].min()
    range_width = max_high - min_low
    
    longest_period = 0
    current_period = 0
    start_idx = None
    longest_start_date = None
    longest_end_date = None

    for i in range(1, len(df)):
        if df['High'].iloc[i] - df['Low'].iloc[i] <= (max_high - min_low) * threshold:
            if current_period == 0:
                start_idx = i
            current_period += 1
        else:
            if current_period > longest_period:
                longest_period = current_period
                longest_start_date = df.index[start_idx]
                longest_end_date = df.index[i-1]
            current_period = 0

    if current_period > longest_period:
        longest_period = current_period
        longest_start_date = df.index[start_idx]
        longest_end_date = df.index[-1]

    return longest_period, range_width, longest_start_date, longest_end_date

def pareto_frontier(currency_pairs, start_date, end_date, min_period=10, min_range_width=0.1):
    results = []
    data = yf.download(currency_pairs, start=start_date, end=end_date)

    for pair in currency_pairs:
        df = data.xs(pair, level=1, axis=1)
        longest_period, range_width, start_period, end_period = calculate_range_period(df)
        # 最低ラインの条件を満たすものだけを結果に追加
        if longest_period >= min_period and range_width >= min_range_width:
            results.append((pair, longest_period, range_width, start_period, end_period))
    
    pareto_set = []
    for i, (pair_i, period_i, width_i, start_i, end_i) in enumerate(results):
        dominated = False
        for j, (pair_j, period_j, width_j, start_j, end_j) in enumerate(results):
            if i != j:
                if (period_j > period_i and width_j <= width_i) or (period_j >= period_i and width_j < width_i):
                    dominated = True
                    break
        if not dominated:
            pareto_set.append((pair_i, period_i, width_i, start_i, end_i))
    
    return results, pareto_set

def cluster_and_display(results, pareto_set, num_clusters=3):
    # パレート解とそれ以外のスコアを取得
    pareto_scores = np.array([(period, width) for _, period, width, _, _ in pareto_set])
    non_pareto_scores = np.array([(period, width) for _, period, width, _, _ in results if (period, width) not in [(p[1], p[2]) for p in pareto_set]])

    # スケーリング
    scaler = StandardScaler()
    all_scores = np.vstack([pareto_scores, non_pareto_scores])
    scaled_scores = scaler.fit_transform(all_scores)

    # クラスタリング
    kmeans = KMeans(n_clusters=num_clusters, n_init=20).fit(scaled_scores)
    labels = kmeans.labels_

    # クラスタごとのインデックスを取得
    pareto_labels = labels[:len(pareto_scores)]
    non_pareto_labels = labels[len(pareto_scores):]

    # 結果を表示
    print("Pareto Optimal Pairs:")
    for (pair, period, width, start_date, end_date), label in zip(pareto_set, pareto_labels):
        print(f"{pair}: Longest Period = {period}, Range Width = {width}, Period Range = {start_date} to {end_date} (Cluster {label})")

    print("\nNon-Pareto Pairs:")
    for (pair, period, width, start_date, end_date), label in zip(results, non_pareto_labels):
        if (period, width) not in [(p[1], p[2]) for p in pareto_set]:
            print(f"{pair}: Longest Period = {period}, Range Width = {width}, Period Range = {start_date} to {end_date} (Cluster {label})")

if __name__ == "__main__":
    results, pareto_optimal_pairs = pareto_frontier(currency_pairs, start_date, end_date, min_period=100, min_range_width=1)
    cluster_and_display(results, pareto_optimal_pairs, num_clusters=3)
