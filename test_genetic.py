import numpy as np
from kabu_backtest import traripi_backtest, data, initial_funds, grid_start, grid_end, entry_intervals, total_thresholds, strategies
from kabu_swap import get_html, parse_swap_points, rename_swap_points, SwapCalculator
from sklearn.model_selection import KFold, train_test_split

url = 'https://fx.minkabu.jp/hikaku/moneysquare/spreadswap.html'
html = get_html(url)
#print(html[:1000])  # デバッグ出力：取得したHTMLの先頭部分を表示
swap_points = parse_swap_points(html)
swap_points = rename_swap_points(swap_points)
#print(swap_points)
calculator = SwapCalculator(swap_points)

entry_interval = entry_intervals[0]
total_threshold = total_thresholds[0]

# 適応度関数の定義 (ここでは f(x1, x2, x3) = -x1^2 + 4*x1 - x2^2 + 2*x2 - x3^2 + 3*x3 を使用)
def fitness_function(data_subset,individual):
    num_trap, profit_width, order_size, strategy, density = individual
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
    return population[selected_indices]


def categorical_to_number(list):

    if list[3]=="long_only":
        list[3] = 0
    if list[3]== "short_only":
        list[3] = 1
    if list[3] == "half_and_half":
        list[3] = 2
    if list[3] == "diamond":
        list[3] = 3
    return list

def number_to_categorical(list):
    if round(list[3]) == 0:
        list[3] = "long_only"
    if round(list[3]) == 1:
        list[3] = "short_only"
    if round(list[3]) == 2:
        list[3] ="half_and_half"
    if round(list[3]) == 3:
        list[3] = "diamond"
    return list

# 交叉
def crossover(parent1, parent2):
    parent1 = categorical_to_number(parent1)
    parent2 = categorical_to_number(parent2)
    alpha = np.random.rand()
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = alpha * parent2 + (1 - alpha) * parent1
    child1 = number_to_categorical(child1)
    child2 = number_to_categorical(child2)
    return child1, child2

# 突然変異
def mutate(individual, mutation_rate, params):
    new_individual = individual.copy()
    for i, (param_name, param_info) in enumerate(params.items()):
        if np.random.rand() < mutation_rate:
            if param_info['type'] == 'int':
                # 整数パラメータの突然変異
                mutation_amount = np.random.choice([-1, 0, 1])
                new_value = individual[i] + mutation_amount
                new_value = np.clip(new_value, param_info['low'], param_info['high'])
                new_individual[i] = int(new_value)
            elif param_info['type'] == 'float':
                # 浮動小数点数パラメータの突然変異
                mutation_amount = np.random.uniform(-0.1, 0.1)
                new_value = individual[i] + mutation_amount
                new_value = np.clip(new_value, param_info['low'], param_info['high'])
                new_individual[i] = new_value
            elif param_info['type'] == 'categorical':
                # カテゴリカルパラメータの突然変異
                choices = param_info['choices']
                current_value = individual[i]
                new_value = np.random.choice([choice for choice in choices if choice != current_value])
                new_individual[i] = new_value
    return new_individual


# メインアルゴリズム
def genetic_algorithm(params,data_subset,pop_size, num_generations, mutation_rate):
    population = create_population(pop_size, params)
    
    for generation in range(num_generations):
        fitnesses = np.array([fitness_function(data_subset,ind) for ind in population])
        print(f"Generation {generation}: Best Fitness = {np.max(fitnesses)}")
        
        selected = select_population(population, fitnesses, pop_size // 2)
        next_population = []
        
        while len(next_population) < pop_size:
            parent1, parent2 = selected[np.random.choice(len(selected), 2, replace=False)]
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            next_population.append(child1)
            next_population.append(child2)
        
        population = np.array(next_population[:pop_size])
    
    best_individual = population[np.argmax(fitnesses)]
    return best_individual


# 遺伝的アルゴリズムの実行
def kfold_cross_validation_genetic(params, X, k, pop_size, num_generations, mutation_rate):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    best_individuals = []
    best_scores = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        
        best_individual = genetic_algorithm(
            params,X_train, pop_size, num_generations, mutation_rate
        )
        best_individuals.append(best_individual)
        
        # 検証データでの評価
        fitness_val = fitness_function(X_val, best_individual)
        best_scores.append(fitness_val)
    
    return best_individuals, best_scores

def main(params, hyper_params):
    X_train_full, X_test_full = train_test_split(data, test_size=0.2, random_state=42)

    best_individuals, best_scores = kfold_cross_validation_genetic(params, X_train_full, hyper_params["k"], hyper_params["pop_size"], hyper_params["num_generations"], hyper_params["mutation_rate"])

    # 最適個体の選択
    best_individual = best_individuals[np.argmax(best_scores)]
    print("Best individual found: ", best_individual)
    print("Best validation score: ", max(best_scores))
    # 最適個体を使用して最終モデルの学習と評価
    best_params = best_individual
    effective_margin_test = fitness_function(X_test_full, best_params)
    print("Test effective margin: {:.2f}".format(effective_margin_test))

    return effective_margin_test



if __name__ == "__main__":

    # パラメータの定義
    params = {
        'num_trap': {'type': 'int', 'low': 1, 'high': 101},
        'profit_width': {'type': 'float', 'low': 0.001, 'high': 100},
        'order_size': {'type': 'categorical', 'choices': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]},
        'strategy': {'type': 'categorical', 'choices': ['long_only', 'short_only', 'half_and_half','diamond']},
        'density':{'type': 'float', 'low':0.1, 'high':10}
    }

    hyper_params = {"k":10,"pop_size":50,"num_generations":10,"mutation_rate":0.2}

    main(params, hyper_params)






