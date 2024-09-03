import optuna
import numpy as np
from sklearn.model_selection import KFold
from kabu_swap import get_html, parse_swap_points, rename_swap_points, SwapCalculator
from kabu_backtest import traripi_backtest, pair, initial_funds, grid_start, grid_end, entry_intervals, total_thresholds, data

entry_interval = entry_intervals
total_threshold = total_thresholds

url = 'https://fx.minkabu.jp/hikaku/moneysquare/spreadswap.html'
html = get_html(url)
swap_points = parse_swap_points(html)
swap_points = rename_swap_points(swap_points)
calculator = SwapCalculator(swap_points)

# 交差検証のfold数
n_splits = 10

# wrapped_objective_functionの定義
def wrapped_objective_function(params, train_data):
    num_trap, profit_width, order_size, strategy, density = params
    
    # traripi_backtestをtrain_dataに対して実行し、effective_marginを計算
    effective_margin, _, _, _, _, _, _, _, _, _ = traripi_backtest(
        calculator, train_data, initial_funds, grid_start, grid_end, num_trap, profit_width, order_size,
        entry_interval, total_threshold, strategy, density
    )
    
    return effective_margin

# 交差検証用のObjective関数
def objective(trial, train_data):
    # パラメータの範囲を定義
    num_trap = trial.suggest_int('num_trap', 1, 100)
    profit_width = trial.suggest_float('profit_width', 0.001, 10.0)
    order_size = trial.suggest_int('order_size', 1, 10) * 1000
    strategy = trial.suggest_categorical('strategy', ['half_and_half', 'diamond', 'long_only', 'short_only'])
    density = trial.suggest_float('density', 1.0, 10.0)

    params = [num_trap, profit_width, order_size, strategy, density]
    
    # train_dataに対してwrapped_objective_functionを実行してスコアを計算
    return wrapped_objective_function(params, train_data)

# 交差検証の実行
kf = KFold(n_splits=n_splits)
best_test_score = -np.inf
best_test_params = None

for train_index, test_index in kf.split(data):
    train_data = data[train_index]
    test_data = data[test_index]

    # Optunaでtrain_dataに対するパラメータ最適化を実行
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, train_data), n_trials=100)

    # best_paramsを使用してtest_dataに対する評価を行う
    best_params = study.best_params
    test_margin = wrapped_objective_function([
        best_params['num_trap'],
        best_params['profit_width'],
        best_params['order_size'],
        best_params['strategy'],
        best_params['density']
    ], test_data)
    
    # テストスコアがこれまでのベストを上回った場合、更新する
    if test_margin > best_test_score:
        best_test_score = test_margin
        best_test_params = best_params

# 最も良いテストスコアとそれに対応するパラメータを表示
print(f'Best test score: {best_test_score}')
print(f'Best test params: {best_test_params}')
