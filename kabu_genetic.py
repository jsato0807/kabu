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
    'density': (1.0, 10.0),
    'strategy': ("long_only", "short_only", "half_and_half", "diamond")
}

# 適応度を最大化する設定
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# 個体を生成する関数
def create_individual():
    return creator.Individual([
        random.randint(*PARAM_BOUNDS['num_trap']),            # num_trap
        random.uniform(*PARAM_BOUNDS['profit_width']),        # profit_width
        random.choice(range(PARAM_BOUNDS['order_size'][0], PARAM_BOUNDS['order_size'][1]+1, 1000)), # order_size
        random.uniform(*PARAM_BOUNDS['density']),             # density
        random.choice(PARAM_BOUNDS['strategy'])               # strategy
    ])

# 個体を変異させる関数
def mutate_individual(individual):
    if random.random() < 0.2:
        individual[0] = random.randint(*PARAM_BOUNDS['num_trap'])
    if random.random() < 0.2:
        individual[1] = random.uniform(*PARAM_BOUNDS['profit_width'])
    if random.random() < 0.2:
        individual[2] = random.choice(range(PARAM_BOUNDS['order_size'][0], PARAM_BOUNDS['order_size'][1]+1, 1000))
    if random.random() < 0.2:
        individual[3] = random.uniform(*PARAM_BOUNDS['density'])
    if random.random() < 0.2:
        individual[4] = random.choice(PARAM_BOUNDS['strategy'])
    return (individual,)

# 個体を交叉させる関数
def crossover_individual(ind1, ind2):
    tools.cxUniform(ind1, ind2, 0.5)
    return ind1, ind2

# 個体の評価関数
def evaluate_individual(individual, data_subset):
    effective_margin, _, _, _, _, _, _, _, _, _ = traripi_backtest(
        calculator, data_subset, initial_funds, grid_start, grid_end, 
        individual[0], individual[1], individual[2], None, None, 
        individual[4], individual[3]
    )
    return (effective_margin,)

# DEAPツールボックスの設定
toolbox = base.Toolbox()
toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", crossover_individual)
toolbox.register("mutate", mutate_individual)
toolbox.register("select", tools.selTournament, tournsize=3)

# 交差検証の実行
def cross_validation(num_folds=5):
    kf = KFold(n_splits=num_folds)
    
    results = []
    best_individuals = []
    
    for train_index, test_index in kf.split(data):
        train_data, test_data = data[train_index], data[test_index]
        
        # 適応度関数をラップしてtrain_dataを利用するように設定
        def wrapped_evaluate(individual):
            return evaluate_individual(individual, train_data)
        
        toolbox.register("evaluate", wrapped_evaluate)
        
        # 遺伝的アルゴリズムの実行
        population = toolbox.population(n=100)
        ngen = 50
        cxpb = 0.5
        mutpb = 0.2

        algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, 
                            stats=None, halloffame=None, verbose=False)

        # テストデータでの最適な個体の評価
        best_individual = tools.selBest(population, k=1)[0]
        test_score = evaluate_individual(best_individual, test_data)
        results.append(test_score[0])
        best_individuals.append(best_individual)
    
    # 各分割の結果を表示
    print("Cross-validation results:", results)
    print("best individuals:", best_individuals)
    print("Average effective margin:", np.mean(results))

if __name__ == "__main__":
    cross_validation()
