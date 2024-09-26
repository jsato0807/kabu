import numpy as np
from deap import base, creator, tools, gp, algorithms
from datetime import datetime
from kabu_backtest import traripi_backtest, fetch_currency_data
from kabu_swap import get_html, parse_swap_points, rename_swap_points, SwapCalculator
import operator
import multiprocessing
import pandas as pd

pair = "AUDNZD=X"

# 外部データの取得と設定
url = 'https://fx.minkabu.jp/hikaku/moneysquare/spreadswap.html'
html = get_html(url)
swap_points = parse_swap_points(html)
swap_points = rename_swap_points(swap_points)
calculator = SwapCalculator(swap_points, pair)

# サンプル取引データ（train_dataとtest_dataに分割）
end_date = datetime.strptime("2024-01-01", "%Y-%m-%d")
start_date = datetime.strptime("2019-01-01", "%Y-%m-%d")
data = fetch_currency_data(pair, start_date, end_date, "1d")
train_data = data[:len(data) // 2]
test_data = data[len(data) // 2:]

# データを分割する
def cpcv(data, k_folds=10, validation_size=4):
    n = len(data)
    fold_size = n // k_folds
    folds = [data[i * fold_size: (i + 1) * fold_size] for i in range(k_folds)]
    
    # バリデーション用のインデックスをランダムに選択
    validation_indices = np.random.choice(range(k_folds), size=validation_size, replace=False)
    
    # バリデーションデータとトレーニングデータを取得
    validation_data = np.concatenate([folds[i] for i in validation_indices])
    train_data = np.concatenate([folds[i] for i in range(k_folds) if i not in validation_indices])
    
    return train_data, validation_data

# 移動平均線を計算する関数
def calculate_moving_average(series: pd.Series, window: int) -> pd.Series:
    moving_average = series.rolling(window=window).mean()
    return moving_average

# 固定された取引パラメータ（初期設定）
initial_funds = 200000
grid_start = 1.02
grid_end = 1.14
entry_interval = 0
total_threshold = 0

# パラメータの範囲を設定
param_ranges = {
    "num_traps": (4, 101),
    "profit_width": (0.01, 50.0),
    "order_size": (1, 10),
    "strategy": ["long_only", "short_only", "half_and_half", "diamond"],  # 仮のストラテジー名
    "density": (0.1, 10.0),
}

# ラムダ式の代わりに通常の関数として定義
def rand101():
    return np.random.randint(-1, 2)

def safe_div(x, y):
    if y == 0:
        return 0  # ゼロ除算の場合、0を返す
    return x / y

def safe_mul(x, y):
    result = x * y
    if np.abs(result) > 1e6:  # 上限を1e6に設定
        return 1e6 if result > 0 else -1e6
    return result

# DEAPの設定
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

pset = gp.PrimitiveSet("MAIN", 4)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(safe_mul, 2)
pset.addPrimitive(safe_div, 2)
pset.addEphemeralConstant("rand101", rand101)

# DEAPのツールボックスを定義
toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("compile", gp.compile, pset=pset)

# パラメータを含む個体を生成
def create_individual():
    params = [
        np.random.randint(*param_ranges["num_traps"]),
        np.random.uniform(*param_ranges["profit_width"]),
        np.random.randint(*param_ranges["order_size"]) * 1000,
        np.random.choice(param_ranges["strategy"]),
        np.random.uniform(*param_ranges["density"]),
    ]
    individual = creator.Individual(toolbox.expr())
    individual.params = params
    return individual

def evaluate(individual, data):
    func = toolbox.compile(expr=individual)
    params = individual.params
    num_traps, profit_width, order_size, strategy, density = params
    
    # 演算子の数が制限を超えた場合、ペナルティを与える
    max_operators = 6  # 演算子の最大数
    operator_count = sum(1 for node in individual if isinstance(node, gp.Primitive))
    
    if operator_count > max_operators:
        return -1e12,  # 評価値として -無限大を返す

    random_window = np.random.randint(1, 100)
    train_data_moving_avg = calculate_moving_average(data, random_window)

    effective_margin, _, realized_profit, _, _, _, _, _, _, _, sharp_ratio, max_draw_down = traripi_backtest(
        calculator, train_data_moving_avg, initial_funds, grid_start, grid_end,
        num_traps, profit_width, order_size, entry_interval,
        total_threshold, strategy=strategy, density=density
    )

    result = func(effective_margin, realized_profit, sharp_ratio, max_draw_down)
    return result,

# パラメータの突然変異
def mutate_params(individual):
    param_idx = np.random.randint(0, len(individual.params))
    if param_idx == 0:
        individual.params[param_idx] = np.random.randint(*param_ranges["num_traps"])
    elif param_idx == 1:
        individual.params[param_idx] = np.random.uniform(*param_ranges["profit_width"])
    elif param_idx == 2:
        individual.params[param_idx] = np.random.randint(*param_ranges["order_size"]) * 1000
    elif param_idx == 3:
        individual.params[param_idx] = np.random.choice(param_ranges["strategy"])
    elif param_idx == 4:
        individual.params[param_idx] = np.random.uniform(*param_ranges["density"])
    return individual,

# 評価関数とパラメータの両方を突然変異させるカスタム関数
def custom_mutate(individual):
    if np.random.rand() < 0.5:
        mutated_expr = gp.mutUniform(individual, expr=toolbox.expr, pset=pset)[0]
        if mutated_expr is None:
            print("mutated_expr is None")
        else:
            # 個別に代入
            for i in range(len(individual)):
                individual[i] = mutated_expr[i]  # 各要素を個別に更新 
    mutate_params(individual)
    return individual,

# 交叉関数のカスタム実装
def custom_mate(ind1, ind2):
    gp.cxOnePoint(ind1, ind2)
    for i in range(len(ind1.params)):
        if np.random.rand() < 0.5:
            ind1.params[i], ind2.params[i] = ind2.params[i], ind1.params[i]
    return ind1, ind2

# Early Stoppingのための関数
def early_stopping(fitness_history, patience=50):
    if len(fitness_history) < patience:
        return False
    return all(fitness_history[-patience] >= score for score in fitness_history[-patience + 1:])

if __name__ == "__main__":
    early_stopping_flag = False
    # その他の設定
    toolbox.register("evaluate", evaluate, data=train_data)  # train_dataを使用
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", custom_mate)
    toolbox.register("mutate", custom_mutate)

    # 遺伝アルゴリズムの進化の過程
    population = [create_individual() for _ in range(100)]
    hof = tools.HallOfFame(1)  # ベスト個体を保存するホールオブフェーム
    fitness_history = []
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:   
        for gen in range(2000):
            # トレーニングデータのサブセットとバリデーションデータを取得
            train_subset, validation_data = cpcv(train_data)

            # 各個体に対して評価を実行
            fitnesses = pool.map(toolbox.evaluate, population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit


            # バリデーションデータに対するスコアを計算
            validation_scores = list(map(lambda ind: evaluate(ind, validation_data), population))


            # 最良個体をハルオブフェームに追加
            hof.update(population)


            # Early stoppingのチェック
            if early_stopping(fitness_history):
                print(f"Early stopping at generation {gen}")
                break

            # 新しい世代の選択
            offspring = toolbox.select(population, len(population), key=lambda ind: validation_scores[population.index(ind)])
            offspring = list(map(toolbox.clone, offspring))

            # 交叉と突然変異を適用
            pairs = zip(offspring[::2], offspring[1::2])
            pool.starmap(toolbox.mate, pairs)

            mutate_offspring = pool.map(toolbox.mutate, offspring)

            # 新しい世代を更新
            population[:] = mutate_offspring


            # ベスト個体のスコアを記録
            best_fitness = hof[0].fitness.values[0]
            fitness_history.append(best_fitness)

            # Early Stoppingのチェック
            if early_stopping(fitness_history):
                early_stopping_flag = True
                break

        test_score = evaluate(hof[0], test_data)
        print(f"テストデータでの評価スコア: {test_score}")


    if early_stopping_flag:
        print("Early stopping was executed")
    print(f"Best individual: {hof[0]}, fitness: {hof[0].fitness.values}")
    print("Parameters:", hof[0].params)  # パラメータ
