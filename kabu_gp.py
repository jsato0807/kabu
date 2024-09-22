import numpy as np
from deap import base, creator, tools, gp, algorithms
from datetime import datetime
from kabu_backtest import traripi_backtest, fetch_currency_data
from kabu_swap import get_html, parse_swap_points, rename_swap_points, SwapCalculator
import operator
import multiprocessing

pair = "AUDNZD=X"

# 外部データの取得と設定
url = 'https://fx.minkabu.jp/hikaku/moneysquare/spreadswap.html'
html = get_html(url)
swap_points = parse_swap_points(html)
swap_points = rename_swap_points(swap_points)
calculator = SwapCalculator(swap_points,pair)

# サンプル取引データ（train_dataとtest_dataに分割）
end_date = datetime.strptime("2024-01-01", "%Y-%m-%d")
start_date = datetime.strptime("2019-01-01", "%Y-%m-%d")
data = fetch_currency_data(pair, start_date, end_date, "1d")
train_data = data[:len(data) // 2]
test_data = data[len(data) // 2:]

# 固定された取引パラメータ（初期設定）
initial_funds = 200000
grid_start = 1.02
grid_end = 1.14
entry_interval = 0
total_threshold = 0

# パラメータの範囲
param_ranges = {
    "num_traps": [4, 101],
    "profit_width": [0.01, 50.0],
    "order_size": [1, 10],
    "strategy": ["long_only", "short_only", "half_and_half", "diamond"],
    "density": [0.01, 10.0]
}


# ラムダ式の代わりに通常の関数として定義
def rand101():
    return np.random.randint(-1, 2)


def safe_div(x, y):
    if y == 0:
        return 1  # ゼロ除算の場合、1を返す
    return x / y


# DEAPの設定
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

pset = gp.PrimitiveSet("MAIN", 4)  # effective_margin, realized_profit, sharp_ratio, max_draw_down の4つを入力
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(safe_div, 2)  # 安全な除算関数を追加
#pset.addPrimitive(np.sin, 1)
#pset.addPrimitive(np.cos, 1)
# 既存コードにおいて rand101 を addEphemeralConstant で使う場合
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
        np.random.randint(*param_ranges["order_size"])*1000,
        np.random.choice(param_ranges["strategy"]),
        np.random.uniform(*param_ranges["density"]),
    ]
    individual = creator.Individual(toolbox.expr())
    individual.params = params  # パラメータを個体に追加
    return individual

# 評価関数
def evaluate(individual, data):
    func = toolbox.compile(expr=individual)  # GPで生成された関数
    params = individual.params  # 個体からパラメータを取得
    num_traps, profit_width, order_size, strategy, density = params
    
    # traripi_backtestの結果を取得
    effective_margin, _, realized_profit, _, _, _, _, _, _, _, sharp_ratio, max_draw_down = traripi_backtest(
        calculator, data, initial_funds, grid_start, grid_end,
        num_traps, profit_width, order_size, entry_interval,
        total_threshold, strategy=strategy, density=density
    )

    # GPで生成された評価関数に全ての結果を入力
    result = func(effective_margin, realized_profit, sharp_ratio, max_draw_down)
    return result,

# パラメータの突然変異
def mutate_params(individual):
    param_idx = np.random.randint(0, len(individual.params))
    if param_idx == 0:  # num_traps
        individual.params[param_idx] = np.random.randint(*param_ranges["num_traps"])
    elif param_idx == 1:  # profit_width
        individual.params[param_idx] = np.random.uniform(*param_ranges["profit_width"])
    elif param_idx == 2:  # order_size
        individual.params[param_idx] = np.random.randint(*param_ranges["order_size"])*1000
    elif param_idx == 3:  # strategy
        individual.params[param_idx] = np.random.choice(param_ranges["strategy"])
    elif param_idx == 4:  # density
        individual.params[param_idx] = np.random.uniform(*param_ranges["density"])
    return individual,


# 評価関数とパラメータの両方を突然変異させるカスタム関数
def custom_mutate(individual):
    # 評価関数の突然変異
    if np.random.rand() < 0.5:  # 50%の確率で評価関数の突然変異
        tools.mutUniform(individual, expr=toolbox.expr, pset=pset)
    
    # パラメータの突然変異
    mutate_params(individual)
    
    return individual,

# 交叉関数のカスタム実装
def custom_mate(ind1, ind2):
    # 評価関数の交叉
    gp.cxOnePoint(ind1, ind2)
    
    # パラメータの交叉（ランダムに親のパラメータを継承）
    for i in range(len(ind1.params)):
        if np.random.rand() < 0.5:
            ind1.params[i], ind2.params[i] = ind2.params[i], ind1.params[i]
    return ind1, ind2


# 進化のプロセス
def evolve_with_train_test():
    pop = [create_individual() for _ in range(100)]
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # 進化を行い、テストデータで評価
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, stats=stats, halloffame=hof, verbose=True)

    best_individual = hof[0]
    # テストデータで最適化された個体を評価
    test_score = evaluate(best_individual, test_data)
    print(f"テストデータでの評価スコア: {test_score}")
    return best_individual

if __name__ == "__main__":

    # 並列処理の設定
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    # その他の設定
    toolbox.register("evaluate", evaluate, data=train_data)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", custom_mate)
    toolbox.register("mutate", custom_mutate)

    best_individual = evolve_with_train_test()

    print("最良の評価関数:", best_individual)
    print("最適化されたパラメータ:", best_individual.params)

    # プールを閉じる
    pool.close()
    pool.join()
