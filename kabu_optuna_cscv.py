import optuna
import numpy as np
import pandas as pd
import scipy.stats as sts
from kabu_swap import get_html, parse_swap_points, rename_swap_points, SwapCalculator
from kabu_backtest import traripi_backtest, pair, initial_funds, grid_start, grid_end, entry_intervals, total_thresholds, data

# パラメータの設定
entry_interval = entry_intervals
total_threshold = total_thresholds

# Swapポイントの取得と計算
url = 'https://fx.minkabu.jp/hikaku/moneysquare/spreadswap.html'
html = get_html(url)
swap_points = parse_swap_points(html)
swap_points = rename_swap_points(swap_points)
calculator = SwapCalculator(swap_points)

# CSCV法によるPBO計算関数
def CSCV(df, n_split, eval=None):
    if eval is None:
        eval = lambda x: np.mean(x) / np.std(x)
    length = len(df)
    X = df.values

    split_idx = [int(length * i / n_split) for i in range(n_split)] + [length]
    group = [list(range(split_idx[i], split_idx[i + 1])) for i in range(n_split)]

    all_split = []
    q = [[i] for i in range(n_split)]
    while q:
        s = q.pop()
        if len(s) == n_split // 2:
            all_split.append(set(s))
            continue
        for i in range(s[-1] + 1, n_split):
            q.append(s + [i])

    w_ary = []
    for spl in all_split:
        insample = []
        outsample = []
        for i in range(n_split):
            if i in spl:
                insample.extend(group[i])
            else:
                outsample.extend(group[i])
        X_is = X[insample, :]
        X_os = X[outsample, :]
        eval_array_is = []
        eval_array_os = []
        for i in range(X.shape[1]):
            eval_array_is.append(eval(X_is[:, i]))
            eval_array_os.append(eval(X_os[:, i]))
        n_star = np.argmax(eval_array_is)
        w = sts.rankdata([-x for x in eval_array_os])[n_star]
        w /= X.shape[1]
        w_ary.append(w)
    
    pbo = len([x for x in w_ary if x < 0.5]) / len(w_ary)
    return pbo

# OptunaのObjective関数
def wrapped_objective_function(params, train_data):
    num_trap, profit_width, order_size, strategy, density = params
    
    effective_margin, _, _, _, _, _, _, _, _, _ = traripi_backtest(
        calculator, train_data, initial_funds, grid_start, grid_end, num_trap, profit_width, order_size,
        entry_interval, total_threshold, strategy, density
    )
    
    return effective_margin

# パラメータ最適化のObjective関数
def objective(trial):
    num_trap = trial.suggest_int('num_trap', 1, 100)
    profit_width = trial.suggest_float('profit_width', 0.001, 10.0)
    order_size = trial.suggest_int('order_size', 1, 10) * 1000
    strategy = trial.suggest_categorical('strategy', ['half_and_half', 'diamond', 'long_only', 'short_only'])
    density = trial.suggest_float('density', 1.0, 10.0)

    params = [num_trap, profit_width, order_size, strategy, density]
    
    # 全データに対してwrapped_objective_functionを実行してスコアを計算
    return wrapped_objective_function(params, data)

# Optunaの実行
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# 最良のパラメータを取得
best_params = study.best_params
print(f'Best parameters: {best_params}')

# CSCVによる評価
df = pd.DataFrame(data)
best_params_list = [
    best_params['num_trap'],
    best_params['profit_width'],
    best_params['order_size'],
    best_params['strategy'],
    best_params['density']
]
pbo_result = CSCV(df, n_split=10, eval=lambda x: wrapped_objective_function(best_params_list, x))

print(f'PBO result: {pbo_result}')
