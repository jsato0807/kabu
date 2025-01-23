import optuna
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import scipy.stats as sts
from kabu_swap import get_html, parse_swap_points, rename_swap_points, SwapCalculator
from kabu_backtest import fetch_currency_data, traripi_backtest, pair, initial_funds, grid_start, grid_end, entry_intervals, total_thresholds, data

# パラメータの設定
entry_interval = entry_intervals
total_threshold = total_thresholds

# Swapポイントの取得と計算
url = 'https://fx.minkabu.jp/hikaku/moneysquare/spreadswap.html'
html = get_html(url)
swap_points = parse_swap_points(html)
swap_points = rename_swap_points(swap_points)
calculator = SwapCalculator(swap_points)

# 最適化とエフェクティブマージン計算関数
def optimize_and_compute_effective_margin(n_trials, train_data):
    def objective(trial):
        num_trap = trial.suggest_int('num_trap', 1, 100)
        profit_width = trial.suggest_float('profit_width', 0.001, 10.0)
        order_size = trial.suggest_int('order_size', 1, 10) * 1000
        strategy = trial.suggest_categorical('strategy', ['half_and_half', 'diamond', 'long_only', 'short_only'])
        density = trial.suggest_float('density', 1.0, 10.0)
        
        params = [num_trap, profit_width, order_size, strategy, density]
        effective_margin = wrapped_objective_function(params, train_data)
        return effective_margin

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    # 各試行の結果を取得
    params_trials = [trial.params for trial in study.trials]
    margin_trials = [trial.value for trial in study.trials]

    return np.array(params_trials), np.array(margin_trials)

# エフェクティブマージンの行列を計算する関数
def compute_effective_margin_matrix(period, n_trials, data):
    n_periods = len(data) // period
    all_param_trials = []
    all_margin_values = []
    periods = []

    for i in range(n_periods):
        start_idx = i * period
        end_idx = (i + 1) * period
        period_data = data[start_idx:end_idx]
        
        # 各期間での最適化とエフェクティブマージンの計算
        best_params, best_margin = optimize_and_compute_effective_margin(n_trials, period_data)
        
        all_param_trials.extend(best_params)
        all_margin_values.extend(best_margin)

    # 各期間の結果を結合して (T, N) 形状にする
    margin_matrix = np.array(all_margin_values).reshape(n_periods, n_trials)
    param_trials_matrix = np.array(all_param_trials).reshape(n_periods, n_trials, -1)  # ここで-1はパラメータ数

    return margin_matrix, param_trials_matrix

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

# `wrapped_objective_function` の定義
def wrapped_objective_function(params, train_data):
    num_trap, profit_width, order_size, strategy, density = params
    effective_margin, _, _, _, _, _, _, _, _, _ = traripi_backtest(
        calculator, train_data, initial_funds, grid_start, grid_end, num_trap, profit_width, order_size,
        entry_interval, total_threshold, strategy, density
    )
    return effective_margin

if __name__ == "__main__":
    interval = "1d"
    end_date = datetime.strptime("2022-01-01", "%Y-%m-%d")  # datetime.now() - timedelta(days=7)
    start_date = datetime.strptime("2010-01-01", "%Y-%m-%d")  # datetime.now() - timedelta(days=14)
    data = fetch_currency_data(pair, start_date, end_date, interval)

    # メインコード
    period = 10  # 例: 10日ごと
    n_trials = 500  # 例: 試行回数
    margin_matrix, param_trials = compute_effective_margin_matrix(period, n_trials, data)

    # 各期間ごとのパラメータとマージン値を結合してDataFrameに変換
    df_matrix = pd.DataFrame(margin_matrix)

    # ベスト5サンプルの抽出と表示
    top_n = 5
    flat_indices = np.argsort(margin_matrix.flatten())[-top_n:]  # すべての値をフラット化してソート
    top_indices = np.unravel_index(flat_indices, margin_matrix.shape)  # 元の形状に戻す

    print("Top 5 Effective Margin Samples:")
    for i, idx in enumerate(zip(*top_indices)):
        period_idx, trial_idx = idx
        margin_value = margin_matrix[period_idx, trial_idx]
        params = param_trials[period_idx, trial_idx]
        start_idx = int(period_idx * period)
        end_idx = int((period_idx + 1) * period)
        date_range = f"{start_date + timedelta(days=start_idx)} to {start_date + timedelta(days=end_idx - 1)}"
        
        print(f"Sample {i + 1}:")
        print(f"  Effective Margin: {margin_value}")
        print(f"  Parameters: {params}")
        print(f"  Data Period: {date_range}")
    
    # CSCVの実行
    n_splits = 10
    pbo = CSCV(df_matrix, n_split=n_splits)

    print(f'CSCV PBO: {pbo}')

