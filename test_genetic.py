import numpy as np
from kabu_backtest import traripi_backtest, data, initial_funds, grid_start, grid_end, entry_intervals, total_thresholds
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
    num_trap, profit_width, order_size, density, strategy = individual
    effective_margin, _, _, _, _, _, _, _, _, _ = traripi_backtest(
        calculator, data_subset, initial_funds, grid_start, grid_end, num_trap, profit_width, order_size, entry_interval, total_threshold, strategy, density
    )
    return effective_margin

# 個体の生成 (実数値ベクトル)
def create_individual(lower_bounds, upper_bounds):
    return np.array([np.random.uniform(lower, upper) for lower, upper in zip(lower_bounds, upper_bounds)])

# 初期集団の生成
def create_population(size, lower_bounds, upper_bounds):
    return np.array([create_individual(lower_bounds, upper_bounds) for _ in range(size)])

# 選択
def select_population(population, fitnesses, num_individuals):
    selected_indices = np.argsort(fitnesses)[-num_individuals:]
    return population[selected_indices]

# 交叉
def crossover(parent1, parent2):
    alpha = np.random.rand()
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = alpha * parent2 + (1 - alpha) * parent1
    return child1, child2

# 突然変異
def mutate(individual, mutation_rate, lower_bounds, upper_bounds):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] += np.random.uniform(-1, 1)
            # 境界を超えないように制限
            individual[i] = np.clip(individual[i], lower_bounds[i], upper_bounds[i])
    return individual

# メインアルゴリズム
def genetic_algorithm(data_subset,pop_size, num_generations, mutation_rate, lower_bounds, upper_bounds):
    population = create_population(pop_size, lower_bounds, upper_bounds)
    
    for generation in range(num_generations):
        fitnesses = np.array([fitness_function(data_subset,ind) for ind in population])
        print(f"Generation {generation}: Best Fitness = {np.max(fitnesses)}")
        
        selected = select_population(population, fitnesses, pop_size // 2)
        next_population = []
        
        while len(next_population) < pop_size:
            parent1, parent2 = selected[np.random.choice(len(selected), 2, replace=False)]
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate, lower_bounds, upper_bounds)
            child2 = mutate(child2, mutation_rate, lower_bounds, upper_bounds)
            next_population.append(child1)
            next_population.append(child2)
        
        population = np.array(next_population[:pop_size])
    
    best_individual = population[np.argmax(fitnesses)]
    return best_individual


# 遺伝的アルゴリズムの実行
def kfold_cross_validation_genetic(X, k, pop_size, num_generations, mutation_rate, lower_bounds, upper_bounds):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    best_individuals = []
    best_scores = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        
        best_individual = genetic_algorithm(
            X_train, pop_size, num_generations, mutation_rate, lower_bounds, upper_bounds
        )
        best_individuals.append(best_individual)
        
        # 検証データでの評価
        fitness_val = fitness_function(X_val, best_individual)
        best_scores.append(fitness_val)
    
    return best_individuals, best_scores

def main(best_individual,param):
    X_train_full, X_test_full = train_test_split(data, test_size=0.2, random_state=42)

    best_individuals, best_scores = kfold_cross_validation_genetic(X_train_full, param["k"], param["pop_size"], param["num_generations"], param["mutation_rate"], param["lower_bounds"], param["upper_bounds"])

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






