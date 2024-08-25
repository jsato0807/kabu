import yfinance as yf
import pandas as pd
from sklearn.cluster import KMeans
from kabu_checkrange import calculate_range_period
import numpy as np

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


def calculate_score(period, width, max_period, min_period, max_width, min_width, alpha=0.5, beta=-0.5):
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
    
    return pareto_set, scored_results

def cluster_near_pareto(pareto_set, all_results, num_clusters=3):
    # パレート最適解のデータをnumpy配列に変換
    pareto_data = np.array([[period, width] for _, period, width in pareto_set])

    # クラスタリング
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pareto_data)
    labels = kmeans.labels_

    # クラスタリング結果の表示
    clusters = {i: [] for i in range(num_clusters)}
    all_data = {pair: (period, width) for pair, period, width, _ in all_results}  # 修正: 4つ目の値(score)を無視

    for idx, (pair, period, width, score) in enumerate(all_results):
        distances = np.linalg.norm(pareto_data - [period, width], axis=1)
        closest_cluster = kmeans.predict([[period, width]])[0]
        clusters[closest_cluster].append((pair, period, width, distances[closest_cluster]))

    return clusters

def print_results(pareto_set, clusters):
    print("Pareto Optimal Pairs:")
    for pair, period, width in pareto_set:
        print(f"  {pair}: Longest Period = {period}, Range Width = {width}")

    print("\nClusters (Nearest to Pareto Optimal):")
    for cluster_id, members in clusters.items():
        print(f"Cluster {cluster_id}:")
        for pair, period, width, distance in members:
            print(f"  {pair}: Longest Period = {period}, Range Width = {width}, Distance to Pareto = {distance:.2f}")

# Pareto最適解の通貨ペアを探す
pareto_optimal_pairs_scores, all_results = pareto_frontier_with_scores(currency_pairs, start_date, end_date)

# パレート最適解に近い通貨ペアのクラスタリング
clusters = cluster_near_pareto(pareto_optimal_pairs_scores, all_results, num_clusters=3)

# 結果の表示
print_results(pareto_optimal_pairs_scores, clusters)
