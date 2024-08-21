import random
from deap import base, creator, tools, algorithms
import numpy as np
from sklearn.model_selection import KFold
from kabu_swap import get_html, parse_swap_points, rename_swap_points, SwapCalculator
from kabu_backtest import data, traripi_backtest, initial_funds, grid_end, grid_start, entry_intervals, total_thresholds, strategies

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
    'profit_width': (0.1, 10),
    'order_size': (1000, 10000),
    'strategy_idx': tuple(i for i in range(len(strategies))),
    'density': (1.0, 10.0)
}

# カスタム関数: order_sizeを1000刻みで生成
def custom_order_size():
    return random.choice(range(PARAM_BOUNDS['order_size'][0], PARAM_BOUNDS['order_size'][1] + 1, 1000))

# 適応度を評価する関数
def evaluate(individual, data_subset):
    num_trap, profit_width, order_size, density, strategy_idx = individual
    num_trap = int(num_trap)
    strategy_idx = int(strategy_idx)  # インデックスを整数に変換
    if strategy_idx < 0 or strategy_idx >= len(strategies):
        strategy_idx = random.randint(0, len(strategies) - 1)
    strategy = strategies[strategy_idx]  # 整数から文字列に変換
    if strategy != 'diamond':
        density = 1.0  # diamond以外の戦略ではdensityを固定値とする
    effective_margin, _, _, _, _, _, _, _, _, _ = traripi_backtest(
        calculator, data_subset, initial_funds, grid_start, grid_end, num_trap, profit_width, order_size, entry_interval, total_threshold, strategy, density
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
toolbox.register("strategy_idx", random.choice, PARAM_BOUNDS['strategy_idx'])  # 修正箇所
toolbox.register("density", random.uniform, *PARAM_BOUNDS['density'])

# 個体と個体群を生成
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.num_trap, toolbox.profit_width, toolbox.order_size, toolbox.strategy_idx, toolbox.density), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 適応度の評価、交叉、突然変異、選択を定義
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutPolynomialBounded, low=[PARAM_BOUNDS['num_trap'][0], PARAM_BOUNDS['profit_width'][0], PARAM_BOUNDS['order_size'][0], PARAM_BOUNDS['density'][0]],
                 up=[PARAM_BOUNDS['num_trap'][1], PARAM_BOUNDS['profit_width'][1], PARAM_BOUNDS['order_size'][1], PARAM_BOUNDS['density'][1]], eta=0.2, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def genetic_algorithm_with_cv(data, k_folds=10):
    best_individuals = []
    kf = KFold(n_splits=k_folds)

    for train_index, test_index in kf.split(data):
        train_data = data[train_index]
        test_data = data[test_index]
        
        # 遺伝的アルゴリズムのパラメータ
        population = toolbox.population(n=50)
        ngen = 40
        cxpb = 0.5
        mutpb = 0.2

        # 適応度の評価関数をtrain_dataで実行するように変更
        def evaluate_individual(individual):
            return toolbox.evaluate(individual, train_data)

        # 遺伝的アルゴリズムのセットアップ
        for ind in population:
            ind.fitness.values = evaluate_individual(ind)

        # アルゴリズムの実行
        algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=True)

        # 最適な解を取得し、test_dataで評価
        best_individual = tools.selBest(population, k=1)[0]
        best_fitness = toolbox.evaluate(best_individual, test_data)[0]
        best_individuals.append((best_individual, best_fitness))

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
