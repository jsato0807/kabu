import optuna
from kabu_swap import get_html, parse_swap_points, rename_swap_points, SwapCalculator
from kabu_backtest import traripi_backtest,pair, initial_funds, grid_start, grid_end,entry_intervals, total_thresholds, data

entry_interval= entry_intervals
total_threshold = total_thresholds

url = 'https://fx.minkabu.jp/hikaku/moneysquare/spreadswap.html'
html = get_html(url)
#print(html[:1000])  # デバッグ出力：取得したHTMLの先頭部分を表示
swap_points = parse_swap_points(html)
swap_points = rename_swap_points(swap_points)
calculator = SwapCalculator(swap_points)


# wrapped_objective_functionの定義
def wrapped_objective_function(params):
    num_trap, profit_width, order_size, strategy, density = params
    effective_margin, _, _, _, _, _, _, _, _, _ = traripi_backtest(
        calculator, data, initial_funds, grid_start, grid_end, num_trap, profit_width, order_size,
        entry_interval, total_threshold, strategy, density
    )
    return effective_margin

# Optuna用のobjective関数の定義
def objective(trial):
    # パラメータの範囲を定義
    num_trap = trial.suggest_int('num_trap', 1, 100)
    profit_width = trial.suggest_float('profit_width', 0.001, 10.0)
    order_size = trial.suggest_int('order_size', 1, 10) * 1000
    strategy = trial.suggest_categorical('strategy', ['half_and_half', 'diamond', 'long_only', 'short_only'])
    density = trial.suggest_float('density', 1.0, 10.0)

    # パラメータをwrapped_objective_functionに渡して評価
    params = [num_trap, profit_width, order_size, strategy, density]
    return wrapped_objective_function(params)

# OptunaのStudyを作成して最適化を実行
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# 結果を表示
print(f'Best params: {study.best_params}')
print(f'Best value: {study.best_value}')

