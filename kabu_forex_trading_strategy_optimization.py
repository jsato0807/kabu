import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from functools import partial  # partialをインポート
from kabu_backtest import pair, start_date, end_date, interval, fetch_currency_data

# 取引ルールに基づく売買シグナルを生成する関数
def generate_signals(prices, p, h, b, transaction_cost):
    signals = np.zeros(len(prices))
    moving_avg = prices.rolling(window=p).mean()
    
    last_trade = -h  # 最後の取引のインデックス（初期値は-hとして全ての取引を可能にする）
    
    for t in range(p, len(prices)):
        if t - 1 < p:  # `p`より小さいインデックスにアクセスしないようにチェック
            continue
        
        # 買いシグナル
        if moving_avg[t] > moving_avg[t-1] and prices[t] - prices[t-1] > b:
            if t - last_trade >= h:  # 最低取引間隔を考慮
                signals[t] = 1
                last_trade = t
        # 売りシグナル
        elif moving_avg[t] < moving_avg[t-1] and prices[t-1] - prices[t] > b:
            if t - last_trade >= h:
                signals[t] = -1
                last_trade = t
        # ホールドシグナル
        else:
            signals[t] = 0
            
    # 取引コストを考慮
    returns = np.diff(prices) / prices[:-1]
    adjusted_returns = signals[:-1] * returns - transaction_cost * np.abs(np.diff(signals))
    return signals, np.sum(adjusted_returns)


# パフォーマンスを評価するための関数
def evaluate_strategy(individual, prices, transaction_cost):
    p, h, b = individual
    _, profit = generate_signals(prices, int(p), int(h), b, transaction_cost)
    return profit,

# 遺伝的アルゴリズムのセットアップ
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int", np.random.randint, 2, 100)  # 移動平均期間p
toolbox.register("attr_int_h", np.random.randint, 1, 10)  # 最低取引間隔h
toolbox.register("attr_float", np.random.uniform, 0.01, 0.05)  # 最小変化閾値b
toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (toolbox.attr_int, toolbox.attr_int_h, toolbox.attr_float), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


# 遺伝的アルゴリズムの実行
def run_ga(prices, transaction_cost, generations=100, pop_size=300):
        # 部分関数を使用してtoolboxに価格と取引コストを渡す
    toolbox.register("evaluate", partial(evaluate_strategy, prices=prices, transaction_cost=transaction_cost))

    population = toolbox.population(n=pop_size)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, verbose=False)

    best_ind = tools.selBest(population, 1)[0]
    return best_ind

# アウト・オブ・サンプルテスト
def out_of_sample_test(train_prices, test_prices, transaction_cost, generations=100, pop_size=300):
    # トレーニング期間でGAを実行
    best_parameters = run_ga(train_prices, transaction_cost, generations, pop_size)
    print("Best parameters (train): p = {}, h = {}, b = {}".format(int(best_parameters[0]), int(best_parameters[1]), best_parameters[2]))
    
    # テスト期間で最適化されたパラメータを適用
    signals, test_profit = generate_signals(test_prices, int(best_parameters[0]), int(best_parameters[1]), best_parameters[2], transaction_cost)
    print("Out-of-sample test profit: {}".format(test_profit))
    
    return best_parameters, test_profit

# データの読み込み（例：EUR/USDの日次終値）
data = fetch_currency_data(pair, start_date, end_date, interval)
prices = pd.DataFrame(data)['Close'].reset_index(drop=True)
# データをトレーニング期間とテスト期間に分ける（例: 80%トレーニング、20%テスト）
train_size = int(0.8 * len(prices))
train_prices = prices[:train_size]
test_prices = prices[train_size:]

# 取引コストの設定（例: 取引ごとに0.001のコスト）
transaction_cost = 0.001

# アウト・オブ・サンプルテストを実行
best_parameters, test_profit = out_of_sample_test(train_prices, test_prices, transaction_cost)
