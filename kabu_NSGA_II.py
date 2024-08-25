import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from deap import base, creator, tools, algorithms
import random
from kabu_checkrange import calculate_range_period

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

def evaluate(individual):
    index = int(individual[0])
    
    # インデックスが範囲内にあるか確認し、補正する
    if index < 0:
        index = 0
    elif index >= len(currency_pairs):
        index = len(currency_pairs) - 1
    
    pair = currency_pairs[index]
    df = data.xs(pair, level=1, axis=1)
    longest_period, range_width = calculate_range_period(df)
    
    # スケーリングの適用
    scores = np.array([[-longest_period, range_width]])
    scaler = StandardScaler()
    scaled_scores = scaler.fit_transform(scores)

    return tuple(scaled_scores[0])

# データを取得
data = yf.download(currency_pairs, start=start_date, end=end_date)

# DEAPでのNSGA-IIの設定
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 0, len(currency_pairs) - 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=len(currency_pairs)-1, eta=1.0, indpb=0.2)

# クラスタリングを用いた選択機能
def custom_select(population, k):
    fitness_values = [ind.fitness.values for ind in population]
    if len(fitness_values) < k:
        return tools.selNSGA2(population, k)
    
    # KMeans クラスタリングの n_init を明示的に設定
    kmeans = KMeans(n_clusters=k, n_init=20).fit(fitness_values)
    labels = kmeans.labels_
    
    # 各クラスタの代表解を選出
    representative_solutions = []
    for cluster_label in set(labels):
        cluster_solutions = [population[i] for i in range(len(population)) if labels[i] == cluster_label]
        representative_solutions.extend(cluster_solutions)
    
    # 代表解から多様性を持たせるために、NSGA2の選択機能を使用
    return tools.selNSGA2(representative_solutions, k)

toolbox.register("select", custom_select)
toolbox.register("map", map)

# 初期化
population_size = 100
mu = 100
lambda_ = min(20, population_size)  # lambda_ を population_size 以下に設定

population = toolbox.population(n=population_size)
algorithms.eaMuPlusLambda(population, toolbox, mu=mu, lambda_=lambda_, cxpb=0.5, mutpb=0.2, ngen=10, verbose=True)

# 結果の出力 (重複を除外)
unique_results = set()
print("Unique Pareto Optimal Pairs:")
for ind in population:
    index = int(ind[0])
    if index < 0 or index >= len(currency_pairs):
        continue
    pair = currency_pairs[index]
    df = data.xs(pair, level=1, axis=1)
    longest_period, range_width = calculate_range_period(df)
    result = f"{pair}: Longest Period = {longest_period}, Range Width = {range_width}"
    if result not in unique_results:
        unique_results.add(result)
        print(result)
