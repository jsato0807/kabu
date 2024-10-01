import numpy as np
from deap import base, creator, tools, gp, algorithms
from datetime import datetime
from kabu_backtest import traripi_backtest, fetch_currency_data
from kabu_swap import get_html, parse_swap_points, rename_swap_points, SwapCalculator
import operator
import multiprocessing
import pandas as pd
import functools
import random

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
train_data = data[len(data) // 2:]
test_data = data[:len(data) // 2]

# データを分割する
def cpcv(data, k_folds=10, validation_size=4):
    n = len(data)
    fold_size = n // k_folds
    folds = [data[i * fold_size: (i + 1) * fold_size] for i in range(k_folds)]
    
    # バリデーション用のインデックスをランダムに選択
    validation_indices = np.random.choice(range(k_folds), size=validation_size, replace=False)
    validation_indices = np.sort(validation_indices)

    train_indices = [i for i in range(k_folds) if i not in validation_indices]
    
    # バリデーションデータとトレーニングデータを取得
    #validation_data = [folds[i] for i in validation_indices]
    #train_data = [folds[i] for i in range(k_folds) if i not in validation_indices]
    
    return train_indices,validation_indices, folds

# 自己相関が高いかどうかを判定するための関数
def is_high_autocorrelation(data1, data2, threshold=0.1):
    combined_data = np.concatenate((data1, data2))  # データを結合して自己相関を計算
    autocorr = pd.Series(combined_data).autocorr(lag=len(data1))  # data1 と data2 の境界部分の自己相関を計算
    return abs(autocorr) >= threshold  # 自己相関が threshold 以上なら高すぎるとみなす

# 境界部分でデータリークを防ぐための前処理
def purging_and_embargo(data, threshold=0.1):
    """
    threshold: 自己相関が大きいとみなすための閾値
    """
    # dataの不連続な部分のインデックスを特定
    for i in range(len(data) - 1):
        # dataの最後のデータと次のdataの最初のデータを切り捨てる
        while len(data[i]) > 0 and len(data[i + 1]) > 0 and is_high_autocorrelation(data[i][-5:], data[i + 1][:5], threshold):
            # train_dataの最後のデータを切り捨て
            data[i] = data[i][:-1]  # 現在のtrain_dataの最後の部分を削除
            data[i + 1] = data[i + 1][1:]  # 次のtrain_dataの最初の部分を削除

    return data


# インデックスに従ってデータを分割する関数
def split_data(folds, indices):
    grouped_data = []
    current_group = [folds[indices[0]]]  # 最初のデータを初期グループに


    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            # 連続している場合はグループに追加
            current_group.append(folds[indices[i]])
        else:
            # 現在のグループの長さによって処理を分ける
            if len(current_group) == 1:
                grouped_data.append(current_group[0])  # 要素が1つならそのまま追加
            else:
                grouped_data.append(pd.concat(current_group))  # 要素が複数なら結合して追加

            current_group = [folds[indices[i]]]  # 新しいグループを初期化

    # 最後のグループを追加
    if len(current_group) == 1:
        grouped_data.append(current_group[0])  # 要素が1つならそのまま追加
    else:
        grouped_data.append(pd.concat(current_group))  # 要素が複数なら結合して追加

    return grouped_data


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

    realized_profit = 0
    required_margin = 0
    position_value = 0
    swap_value = 0
    effective_margin_max = -np.inf
    effective_margin_min = np.inf

    for sequence in data:
        sequence = pd.Series(sequence)

        # sequence が空の場合はスキップ
        if sequence.empty:
            print("Warning: Empty sequence detected, skipping moving average calculation.")
            continue

        # sequence が数値データ型でない場合のチェック
        if not pd.api.types.is_numeric_dtype(sequence):
            print(f"Warning: Non-numeric data detected in sequence: {sequence}, skipping.")
            continue

        random_window = np.random.randint(1, 100)
        train_data_moving_avg = calculate_moving_average(sequence, random_window)

        effective_margin, _, realized_profit, position_value, swap_value, required_margin, _, _, _, sharp_ratio, max_draw_down, effective_margin_max, effective_margin_min = traripi_backtest(
            calculator, train_data_moving_avg, initial_funds, grid_start, grid_end,
            num_traps, profit_width, order_size, entry_interval,
            total_threshold, strategy=strategy, density=density,
            realized_profit=realized_profit,required_margin=required_margin,position_value=position_value,swap_value=swap_value,
            effective_margin_max=effective_margin_max,effective_margin_min=effective_margin_min
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


# カスタムトーナメント選択関数
def custom_tournament_selection(population, validation_data, tournsize):
    selected = []
    for _ in range(len(population)):
        # トーナメント参加者を選ぶ
        aspirants = random.sample(population, tournsize)
        # validation_dataを使ってスコアを計算
        scores = [evaluate(ind, validation_data)[0] for ind in aspirants]
        # スコアが最も高い個体を選ぶ
        winner = aspirants[scores.index(max(scores))]
        selected.append(winner)
    return selected

# Early Stoppingのための関数
def early_stopping(fitness_history, patience=50):
    if len(fitness_history) < patience:
        return False
    return all(fitness_history[-patience] >= score for score in fitness_history[-patience + 1:])

if __name__ == "__main__":
    early_stopping_flag = False
    # その他の設定
    # 選択関数をカスタム関数に登録
    toolbox.register("select", custom_tournament_selection, tournsize=3)
    toolbox.register("mate", custom_mate)
    toolbox.register("mutate", custom_mutate)

    # 遺伝アルゴリズムの進化の過程
    population = [create_individual() for _ in range(100)]
    hof = tools.HallOfFame(1)  # ベスト個体を保存するホールオブフェーム
    fitness_history = []
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:   
        for gen in range(2000):
            # トレーニングデータのサブセットとバリデーションデータを取得
            train_indices, validation_indices, folds = cpcv(train_data)


            #　不連続なデータを独立させるために自己相関の高いデータ点を切り捨て
            train_subset = split_data(folds, train_indices)
            validation_data = split_data(folds, validation_indices)

            train_subset = purging_and_embargo(train_subset)
            validation_data = purging_and_embargo(validation_data)
            

            # evaluate関数にtrain_subsetを渡す
            evaluate_with_data = functools.partial(evaluate, data=train_subset)
            toolbox.register("evaluate_with_data", evaluate_with_data)  # train_dataを使用


            # 各個体に対して評価を実行
            fitnesses = pool.map(toolbox.evaluate_with_data, population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit


            ## バリデーションデータに対するスコアを計算
            #validation_scores = list(map(lambda ind: evaluate(ind, validation_data), population))


            # 最良個体をハルオブフェームに追加
            hof.update(population)


            # Early stoppingのチェック
            if early_stopping(fitness_history):
                print(f"Early stopping at generation {gen}")
                break

            # 新しい世代の選択
            offspring = toolbox.select(population, validation_data)
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
