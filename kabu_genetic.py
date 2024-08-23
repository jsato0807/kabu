import random
from deap import base, creator, tools, algorithms
import numpy as np
from sklearn.model_selection import KFold
from kabu_swap import get_html, parse_swap_points, rename_swap_points, SwapCalculator
from kabu_backtest import data, traripi_backtest, initial_funds, grid_end, grid_start, entry_intervals, total_thresholds, strategies
from sklearn.preprocessing import OneHotEncoder
from functools import partial

entry_interval = entry_intervals[0]
total_threshold = total_thresholds[0]

url = 'https://fx.minkabu.jp/hikaku/moneysquare/spreadswap.html'
html = get_html(url)
swap_points = parse_swap_points(html)
swap_points = rename_swap_points(swap_points)
calculator = SwapCalculator(swap_points)

# パラメータの範囲を定義
PARAM_BOUNDS = {
    'num_trap': (4, 101),
    'profit_width': (0.001, 100),
    'order_size': (1000, 10000),
    'density': (1.0, 10.0)
}

# カスタム関数: order_sizeを1000刻みで生成
def custom_order_size():
    return random.choice(range(PARAM_BOUNDS['order_size'][0], PARAM_BOUNDS['order_size'][1] + 1, 1000))

# ワンホットエンコーディングの準備
encoder = OneHotEncoder(sparse=False)
encoder.fit(np.array(strategies).reshape(-1, 1))

# 適応度を評価する関数
def evaluate(individual, data_subset):
    num_trap, profit_width, order_size, density = individual[:4]
    strategy_onehot = individual[4:]
    strategy_idx = np.argmax(strategy_onehot)  # ワンホットエンコーディングをデコード
    strategy = strategies[strategy_idx]  # インデックスから戦略を取得
    
    if strategy != 'diamond':
        density = 1.0  # diamond以外の戦略ではdensityを固定値とする
    
    effective_margin, _, _, _, _, _, _, _, _, _ = traripi_backtest(
        calculator, data_subset, initial_funds, grid_start, grid_end, int(num_trap), profit_width, order_size, entry_interval, total_threshold, strategy, density
    )
    
    return effective_margin,

# 遺伝的アルゴリズムのセットアップ
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# 各パラメータに対して初期値をランダムに生成
toolbox.register("num_trap", random.randint, *PARAM_BOUNDS['num_trap'])
toolbox.register("profit_width", random.uniform, *PARAM_BOUNDS['profit_width'])
toolbox.register("order_size", custom_order_size)
toolbox.register("density", random.uniform, *PARAM_BOUNDS['density'])

# strategyのワンホットエンコーディング
def random_strategy():
    return encoder.transform([[random.choice(strategies)]])[0]

# 個体と個体群を生成
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.num_trap, toolbox.profit_width, toolbox.order_size, toolbox.density, random_strategy), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 適応度の評価、交叉、突然変異、選択を定義
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutPolynomialBounded, low=[PARAM_BOUNDS['num_trap'][0], PARAM_BOUNDS['profit_width'][0], PARAM_BOUNDS['order_size'][0], PARAM_BOUNDS['density'][0]],
                 up=[PARAM_BOUNDS['num_trap'][1], PARAM_BOUNDS['profit_width'][1], PARAM_BOUNDS['order_size'][1], PARAM_BOUNDS['density'][1]], eta=0.2, indpb=0.2)

# `strategy` の突然変異は別途処理
def mutate_individual(individual):
    # num_trap, profit_width, order_size, density の部分に突然変異を適用
    individual[:4] = tools.mutPolynomialBounded(individual[:4], 
                                                low=[PARAM_BOUNDS['num_trap'][0], PARAM_BOUNDS['profit_width'][0], PARAM_BOUNDS['order_size'][0], PARAM_BOUNDS['density'][0]],
                                                up=[PARAM_BOUNDS['num_trap'][1], PARAM_BOUNDS['profit_width'][1], PARAM_BOUNDS['order_size'][1], PARAM_BOUNDS['density'][1]],
                                                eta=0.2, indpb=0.2)[0]
    
    # strategy の部分に突然変異を適用
    if random.random() < 0.2:
        individual[4:] = random_strategy()  # `strategy` の突然変異
    
    return individual,


toolbox.register("mutate", mutate_individual)
toolbox.register("select", tools.selTournament, tournsize=3)

def genetic_algorithm_with_cv(data, k_folds=10):
    best_individuals = []
    kf = KFold(n_splits=k_folds)

    for train_index, test_index in kf.split(data):
        train_data = data[train_index]
        test_data = data[test_index]
        
        # 適応度関数をラップしてtrain_dataを利用するように設定
        def wrapped_evaluate(individual):
            return evaluate(individual, train_data)
        
        # DEAPのtoolboxにラップした評価関数を登録
        toolbox.register("evaluate", wrapped_evaluate)

        # 遺伝的アルゴリズムのパラメータ
        population = toolbox.population(n=50)
        ngen = 40
        cxpb = 0.5
        mutpb = 0.2

        # 遺伝的アルゴリズムのセットアップ
        for ind in population:
            ind.fitness.values = toolbox.evaluate(ind)

        # アルゴリズムの実行
        algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=True)

        # 最適な解を取得し、test_dataで評価
        best_individual = tools.selBest(population, k=1)[0]
        best_fitness = evaluate(best_individual, test_data)[0]
        best_individuals.append((best_individual, best_fitness))

        # evaluate関数の登録を解除
        del toolbox.evaluate

    return best_individuals

# 交差検証で得られた最適な個体を集約
best_individuals = genetic_algorithm_with_cv(data)

# 各分割での最適なパラメータとその評価値を表示
for i, (ind, margin) in enumerate(best_individuals):
    print(f"Fold {i+1} - Best Individual: {ind}")
    print(f"Effective Margin: {margin}")

# 全体の平均を計算
mean_effective_margin = np.mean([margin for _, margin in best_individuals])
print(f"Mean Effective Margin across all folds: {mean_effective_margin}")
