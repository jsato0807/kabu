import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import talib
from sklearn.impute import SimpleImputer
import yfinance as yf
from kabu_backtest import pair, start_date, end_date, interval

def fetch_currency_data(pair, start, end, interval):
    """
    Fetch historical currency pair data from Yahoo Finance.
    """
    data = yf.download(pair, start=start, end=end, interval=interval)
    print(f"Fetched data length: {len(data)}")
    return data

# データの読み込み（例：技術指標や価格データ）
data = fetch_currency_data(pair, start_date, end_date, interval)
data = data.dropna()
# RSIの計算
data['RSI'] = talib.RSI(data['Close'], timeperiod=14)

# その他の技術指標の計算
data['ROC'] = talib.ROC(data['Close'], timeperiod=14)
data['EMA'] = talib.EMA(data['Close'], timeperiod=14)
data['SMA'] = talib.SMA(data['Close'], timeperiod=14)
data['WMA'] = talib.WMA(data['Close'], timeperiod=14)
data['MACD'], _, _ = talib.MACD(data['Close'])

# RSIに基づいてトレンドをラベリング
def rsi_label(row):
    if row['RSI'] > 70:
        return -1  # 過熱（売りシグナル）→ 下降トレンド
    elif row['RSI'] < 30:
        return 1   # 売られ過ぎ（買いシグナル）→ 上昇トレンド
    else:
        return 0   # 横ばいトレンド

data['Market_Label'] = data.apply(rsi_label, axis=1)

print(data[['Close', 'RSI', 'ROC', 'EMA', 'SMA', 'WMA', 'MACD', 'Market_Label']])

# パーセント変化率を計算
data['Return'] = data['Close'].pct_change()

# 閾値を設定してラベリング（例: 1%を閾値に設定）
def return_label(row, threshold=0.01):
    if row['Return'] > threshold:
        return 1   # 上昇トレンド
    elif row['Return'] < -threshold:
        return -1  # 下降トレンド
    else:
        return 0   # 横ばいトレンド

data['Market_Type'] = data.apply(return_label, axis=1)

print(data[['Close', 'Return', 'Market_Type']])

# 特徴量とラベルの分割
X = data[['RSI', 'ROC', 'EMA', 'SMA', 'WMA', 'MACD']]  # 特徴量: 技術指標
y = data['Market_Type']  # ラベル: 市場の種類（上昇、横ばい、下降）

# データの前処理
# NaN 値を補完する
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# スケーリング
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# トレーニングデータとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# NaN 値の確認
print("NaN values in X_train after imputation:")
print(pd.DataFrame(X_train).isna().sum())

print("NaN values in y_train after imputation:")
print(pd.Series(y_train).isna().sum())


# SVMモデルの訓練
clf = svm.SVC(kernel='rbf', C=10, gamma=0.001)
clf.fit(X_train, y_train)

# テストデータで予測
y_pred = clf.predict(X_test)

# 精度の評価
print("Accuracy:", accuracy_score(y_test, y_pred))


import random
from deap import base, creator, tools, algorithms

# 適合度と個体の定義
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# GAのパラメータ設定
IND_SIZE = 6  # RSI, ROC, EMA, SMA, WMA, MACD
toolbox = base.Toolbox()
toolbox.register("individual", tools.initRepeat, creator.Individual, lambda: random.uniform(0, 1), IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 突然変異の定義（動的に変更される）
def dynamic_mutate(individual, mutation_rate):
    tools.mutGaussian(individual, mu=0, sigma=1, indpb=mutation_rate)
    return individual,

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", dynamic_mutate, mutation_rate=0.2)  # 初期突然変異率
toolbox.register("select", tools.selTournament, tournsize=3)

# 評価関数の定義
def evaluate(individual):
    return sum(individual),  # ダミーの評価値、実際には投資リターンなどを計算

toolbox.register("evaluate", evaluate)

# 環境の変化に応じて突然変異率を変更する
def adjust_mutation_rate(gen, env_state):
    if env_state == "volatile":  # 市場が不安定な場合
        return 0.4  # 高い突然変異率（ハイパーミューテーション）
    elif env_state == "stable":  # 市場が安定している場合
        return 0.1  # 低い突然変異率
    return 0.2  # 通常時

# 遺伝的アルゴリズムの実行
def run_dynamic_ga(env_state):
    pop = toolbox.population(n=100)
    CXPB, MUTPB = 0.5, 0.2  # 交叉確率と突然変異確率
    NGEN = 50  # 世代数

    for gen in range(NGEN):
        # 突然変異率を環境に基づいて調整する
        mutation_rate = adjust_mutation_rate(gen, env_state)
        toolbox.register("mutate", dynamic_mutate, mutation_rate=mutation_rate)

        # 次世代の個体群を選択
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # 交叉
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # 突然変異
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # 適合度が無効な個体を評価
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 次世代に更新
        pop[:] = offspring

        # 最適化された個体の出力
        best_individual = tools.selBest(pop, 1)[0]
        print(f"Gen {gen}: Best individual: {best_individual}, Fitness: {best_individual.fitness.values}")

# 環境が安定していると仮定
env_state = "stable"
run_dynamic_ga(env_state)


# SVMで市場を分類
market_type = clf.predict(X_test)  # テストデータに基づく市場の予測

# 各市場タイプに応じて異なるGAを適用
if market_type[0] == 1:  # 上昇トレンド
    print("Bullish Market: Apply GA for bullish strategies.")
    # GAを実行して最適化された戦略を適用
    # (前述のGAコードを呼び出して最適化)

elif market_type[0] == -1:  # 下降トレンド
    print("Bearish Market: Apply GA for bearish strategies.")
    # 下降トレンド用のGA戦略を適用

else:  # 横ばいトレンド
    print("Sideways Market: Apply GA for neutral strategies.")
    # 横ばいトレンド用のGA戦略を適用
