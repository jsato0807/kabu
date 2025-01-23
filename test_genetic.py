import numpy as np
from kabu_backtest import traripi_backtest, data, initial_funds, grid_start, grid_end, entry_intervals, total_thresholds, strategies
from kabu_swap import get_html, parse_swap_points, rename_swap_points, SwapCalculator
from sklearn.model_selection import KFold, train_test_split

# 初期設定は変更ありません
url = 'https://fx.minkabu.jp/hikaku/moneysquare/spreadswap.html'
html = get_html(url)
swap_points = parse_swap_points(html)
swap_points = rename_swap_points(swap_points)
calculator = SwapCalculator(swap_points)

entry_interval = entry_intervals[0]
total_threshold = total_thresholds[0]

# 適応度関数の定義
def fitness_function(data_subset, individual):
    num_trap, profit_width, order_size, strategy, density = individual
    print(individual)
    effective_margin, _, _, _, _, _, _, _, _, _ = traripi_backtest(
        calculator, data_subset, initial_funds, grid_start, grid_end, num_trap, profit_width, order_size, entry_interval, total_threshold, strategy, density
    )
    return effective_margin

# 個体の生成 (実数値ベクトル)
def create_individual(params):
    individual = []
    for param_name, param_info in params.items():
        if param_info['type'] == 'int':
            individual.append(np.random.randint(param_info['low'], param_info['high'] + 1))
        elif param_info['type'] == 'float':
            individual.append(np.random.uniform(param_info['low'], param_info['high']))
        elif param_info['type'] == 'categorical':
            individual.append(np.random.choice(param_info['choices']))
    return individual

# 初期集団の生成
def create_population(size, params):
    return [create_individual(params) for _ in range(size)]

# 選択
def select_population(population, fitnesses, num_individuals):
    selected_indices = np.argsort(fitnesses)[-num_individuals:]
    return [population[i] for i in selected_indices]

# 交叉
def crossover(parent1, parent2, params):
    alpha = np.random.rand()
    child1 = []
    child2 = []

    for i, (param_name, param_info) in enumerate(params.items()):
        if param_info['type'] == 'int':
            # 整数パラメータの場合、整数型で計算
            value1 = int(alpha * parent1[i] + (1 - alpha) * parent2[i])
            value2 = int(alpha * parent2[i] + (1 - alpha) * parent1[i])
            child1.append(value1)
            child2.append(value2)
        elif param_info['type'] == 'float':
            # 浮動小数点数パラメータの場合、そのまま計算
            value1 = alpha * parent1[i] + (1 - alpha) * parent2[i]
            value2 = alpha * parent2[i] + (1 - alpha) * parent1[i]
            child1.append(value1)
            child2.append(value2)
        elif param_info['type'] == 'categorical':
            # カテゴリカルパラメータの場合、1つを選択
            value1 = parent1[i] if np.random.rand() > 0.5 else parent2[i]
            value2 = parent2[i] if np.random.rand() > 0.5 else parent1[i]
            child1.append(value1)
            child2.append(value2)
    
    return child1, child2


# 突然変異
def mutate(individual, mutation_rate, params):
    new_individual = individual.copy()
    for i, (param_name, param_info) in enumerate(params.items()):
        if np.random.rand() < mutation_rate:
            if param_info['type'] == 'int':
                mutation_amount = np.random.choice([-1, 0, 1])
                new_value = individual[i] + mutation_amount
                new_value = np.clip(new_value, param_info['low'], param_info['high'])
                new_individual[i] = int(new_value)
            elif param_info['type'] == 'float':
                mutation_amount = np.random.uniform(-0.1, 0.1)
                new_value = individual[i] + mutation_amount
                new_value = np.clip(new_value, param_info['low'], param_info['high'])
                new_individual[i] = new_value
            elif param_info['type'] == 'categorical':
                choices = param_info['choices']
                current_value = individual[i]
                new_value = np.random.choice([choice for choice in choices if choice != current_value])
                new_individual[i] = new_value
    return new_individual

# メインアルゴリズム
def genetic_algorithm(params, data_subset, pop_size, num_generations, mutation_rate):
    population = create_population(pop_size, params)
    
    for generation in range(num_generations):

        fitnesses = np.array([fitness_function(data_subset, ind) for ind in population])
        print(f"Generation {generation}: Best Fitness = {np.max(fitnesses)}")
        
        selected = select_population(population, fitnesses, pop_size // 2)
        next_population = []
        
        while len(next_population) < pop_size:
            parent1, parent2 = [selected[i] for i in np.random.choice(len(selected), 2, replace=False)]
            child1, child2 = crossover(parent1, parent2, params)
            child1 = mutate(child1, mutation_rate, params)
            child2 = mutate(child2, mutation_rate, params)
            next_population.append(child1)
            next_population.append(child2)
        
        population = next_population[:pop_size]
    
    best_individual = population[np.argmax(fitnesses)]
    return best_individual

# 遺伝的アルゴリズムの実行
def kfold_cross_validation_genetic(params, X, k, pop_size, num_generations, mutation_rate):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    best_individuals = []
    best_scores = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        
        best_individual = genetic_algorithm(params, X_train, pop_size, num_generations, mutation_rate)
        best_individuals.append(best_individual)
        
        fitness_val = fitness_function(X_val, best_individual)
        best_scores.append(fitness_val)
    
    return best_individuals, best_scores

def main(params, hyper_params):
    X_train_full, X_test_full = train_test_split(data, test_size=0.2, random_state=42)

    best_individuals, best_scores = kfold_cross_validation_genetic(params, X_train_full, hyper_params["k"], hyper_params["pop_size"], hyper_params["num_generations"], hyper_params["mutation_rate"])

    best_individual = best_individuals[np.argmax(best_scores)]
    print("Best individual found: ", best_individual)
    print("Best validation score: ", max(best_scores))
    
    best_params = best_individual
    effective_margin_test = fitness_function(X_test_full, best_params)
    print("Test effective margin: {:.2f}".format(effective_margin_test))

    return effective_margin_test

if __name__ == "__main__":
    params = {
        'num_trap': {'type': 'int', 'low': 1, 'high': 101},
        'profit_width': {'type': 'float', 'low': 0.001, 'high': 100},
        'order_size': {'type': 'categorical', 'choices': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]},
        'strategy': {'type': 'categorical', 'choices': ['long_only', 'short_only', 'half_and_half', 'diamond']},
        'density': {'type': 'float', 'low': 0.1, 'high': 10}
    }

    hyper_params = {"k": 10, "pop_size": 10, "num_generations": 50, "mutation_rate": 0.2}

    main(params, hyper_params)

