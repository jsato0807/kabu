import random
import numpy as np
from sklearn.model_selection import KFold
from deap import base, creator, tools, algorithms
from kabu_backtest import traripi_backtest, data, initial_funds, grid_start, grid_end, strategies, entry_intervals, total_thresholds
# 適応度クラスと個体クラスを定義
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# パラメータの範囲を定義
PARAM_BOUNDS = {
    'num_trap': (1, 100),      # 例として1から20の範囲
    'profit_width': (0.1, 10),   # 例として1から10の範囲
    'order_size': (1000, 100000),     # 例として1から10の範囲
    'density': (0.1, 10)         # 例として1から10の範囲
}

STRATEGIES = strategies

entry_interval = entry_intervals[0]
total_threshold = total_thresholds[0]

# 個体を初期化する関数
def create_individual():
    num_trap = random.randint(*PARAM_BOUNDS['num_trap'])
    profit_width = random.uniform(*PARAM_BOUNDS['profit_width'])
    order_size = random.uniform(*PARAM_BOUNDS['order_size'])
    density = random.uniform(*PARAM_BOUNDS['density'])
    strategy_idx = random.randint(0, len(STRATEGIES) - 1)  # 0から3の範囲で整数を選択
    if STRATEGIES[strategy_idx] == 'diamond':
        density = random.uniform(*PARAM_BOUNDS['density'])
    else:
        density = 1.0  # diamond以外の戦略ではdensityを固定値とする
    return creator.Individual([num_trap, profit_width, order_size, density, strategy_idx])


# 適応度を評価する関数
def evaluate(individual, data_subset):
    num_trap, profit_width, order_size, density, strategy_idx = individual
    num_trap = int(num_trap)
    strategy_idx = int(strategy_idx)  # インデックスを整数に変換
    if strategy_idx < 0 or strategy_idx >= len(STRATEGIES):
        strategy_idx = random.randint(0, len(STRATEGIES) - 1)
    strategy = STRATEGIES[strategy_idx]  # 整数から文字列に変換
    if strategy != 'diamond':
        density = 1.0  # diamond以外の戦略ではdensityを固定値とする
    effective_margin, _, _, _, _, _, _, _, _, _ = traripi_backtest(
        data_subset, initial_funds, grid_start, grid_end, num_trap, profit_width, order_size, entry_interval, total_threshold, strategy, density
    )
    
    return effective_margin,

def mate(ind1, ind2):
    # 数値型遺伝子の交叉
    for i in range(len(ind1) - 1):  # strategyは除く
        if random.random() < CROSSOVER_RATE:
            alpha = random.random()
            ind1[i], ind2[i] = alpha * ind1[i] + (1 - alpha) * ind2[i], alpha * ind2[i] + (1 - alpha) * ind1[i]
    
    # strategyの交叉（整数値の交換）
    if random.random() < CROSSOVER_RATE:
        ind1[-1], ind2[-1] = ind2[-1], ind1[-1]
    
    return ind1, ind2


def cross_validate_and_optimize(individual):
        # 遺伝子操作を定義
    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Blend crossover
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)  # Gaussian mutation
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", mate)
    

    # パラメータ設定
    POPULATION_SIZE = 100
    MUTATION_RATE = 0.2
    CROSSOVER_RATE = 0.5
    MAX_GENERATIONS = 500
    INITIAL_THRESHOLD = 5  # 初期閾値
    THRESHOLD_INCREMENT = 5  # 各世代で増加させる閾値の
    
    # 初期集団を生成
    population = toolbox.population(n=POPULATION_SIZE)

    kf = KFold(n_splits=10)
    margins = []
    
    for train_index, test_index in kf.split(data):
        X_train, X_test = data[train_index], data[test_index]
        
        
        # 遺伝的アルゴリズムのメインループ
        for gen in range(MAX_GENERATIONS):
            offspring = algorithms.varAnd(population, toolbox, cxpb=CROSSOVER_RATE, mutpb=MUTATION_RATE)
            fits = [toolbox.evaluate(ind,X_train) for ind in offspring]
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            
            population = toolbox.select(offspring, k=len(population))
            
            top_individual = tools.selBest(population, k=1)[0]
            print(f"Generation: {gen}\tBest: {top_individual}\tFitness: {top_individual.fitness.values[0]}")
        
        
        
            # 適応度の範囲確認
            fitness_values = [ind.fitness.values[0] for ind in population]
            print(f"Fitness Range: Min: {min(fitness_values)}, Max: {max(fitness_values)}")
        
            # 動的な閾値の設定
            current_threshold = INITIAL_THRESHOLD + gen * THRESHOLD_INCREMENT 
            print(f"Current Threshold: {current_threshold}")
            # 終了条件（適応度が一定の値に達したら終了）
            print(f'top_indivisual.fitness.values[0]:{top_individual.fitness.values[0]}, current_threshold:{current_threshold}')
            if top_individual.fitness.values[0] >= current_threshold:  # 適宜設定
                break
        
        best_individual = tools.selBest(population, k=1)[0]
        print(f"Best individual: {best_individual}\tFitness: {best_individual.fitness.values[0]}")
        

        best_num_trap, best_profit_width, best_order_size, best_density, best_strategy_idx = best_individual
        best_strategy = strategies[best_strategy_idx]

        effective_margin, _, _, _, _, _, _, _, _, _ = traripi_backtest(
            X_test, initial_funds, grid_start, grid_end, best_num_trap, best_profit_width, best_order_size,
            entry_interval, total_threshold, best_strategy, best_density
        )
        margins.append(effective_margin)
    
    return np.mean(margins),


# 遺伝子操作を定義
toolbox = base.Toolbox()
toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Blend crossover
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)  # Gaussian mutation
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("cross_validate_and_optimize", cross_validate_and_optimize)
toolbox.register("mate", mate)


# パラメータ設定
POPULATION_SIZE = 100
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.5
MAX_GENERATIONS = 500
INITIAL_THRESHOLD = 5  # 初期閾値
THRESHOLD_INCREMENT = 5  # 各世代で増加させる閾値の

# 初期集団を生成
population = toolbox.population(n=POPULATION_SIZE)

# 遺伝的アルゴリズムのメインループ
for gen in range(MAX_GENERATIONS):
    offspring = algorithms.varAnd(population, toolbox, cxpb=CROSSOVER_RATE, mutpb=MUTATION_RATE)
    fits = map(toolbox.cross_validate_and_optimize, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    
    population = toolbox.select(offspring, k=len(population))
    
    top_individual = tools.selBest(population, k=1)[0]
    print(f"Generation: {gen}\tBest: {top_individual}\tFitness: {top_individual.fitness.values[0]}")



    # 適応度の範囲確認
    fitness_values = [ind.fitness.values[0] for ind in population]
    print(f"Fitness Range: Min: {min(fitness_values)}, Max: {max(fitness_values)}")

    # 動的な閾値の設定
    current_threshold = INITIAL_THRESHOLD + gen * THRESHOLD_INCREMENT 
    print(f"Current Threshold: {current_threshold}")
    # 終了条件（適応度が一定の値に達したら終了）
    print(f'top_indivisual.fitness.values[0]:{top_individual.fitness.values[0]}, current_threshold:{current_threshold}')
    if top_individual.fitness.values[0] >= current_threshold:  # 適宜設定
        break

best_individual = tools.selBest(population, k=1)[0]
print(f"Best individual: {best_individual}\tFitness: {best_individual.fitness.values[0]}")
