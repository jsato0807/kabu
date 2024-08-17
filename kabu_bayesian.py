import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from mpl_toolkits.mplot3d import Axes3D
from skopt.utils import use_named_args
from sklearn.model_selection import KFold
from kabu_backtest import traripi_backtest, data, initial_funds, grid_start, grid_end, entry_intervals, total_thresholds, strategies
import numpy as np
from kabu_swap import get_html, parse_swap_points, rename_swap_points, SwapCalculator

url = 'https://fx.minkabu.jp/hikaku/moneysquare/spreadswap.html'
html = get_html(url)
#print(html[:1000])  # デバッグ出力：取得したHTMLの先頭部分を表示
swap_points = parse_swap_points(html)
swap_points = rename_swap_points(swap_points)
#print(swap_points)
calculator = SwapCalculator(swap_points)


# パラメータ空間の定義
space = [
    Integer(1, 101, name='num_trap'),
    Real(0.1, 100, name='profit_width'),
    Categorical([1000,2000,3000,4000,5000,6000,7000,8000,9000, 10000], name='order_size'),
    Categorical(['long_only', 'short_only', 'half_and_half','diamond'], name='strategy'),
    Real(1.0,10, name='density')
]

# 固定値
default_density = 1.0

entry_interval = entry_intervals[0]
total_threshold = total_thresholds[0]

# 評価関数の定義
def cross_validate_and_optimize(params, space):
    num_trap, profit_width, order_size, strategy = params[:4]
    density = params[4] if len(params) > 4 else default_density

    kf = KFold(n_splits=10)
    margins = []

    for train_index, test_index in kf.split(data):
        X_train, X_test = data[train_index], data[test_index]

        # 内部ベイズ最適化
        def wrapped_objective_function(params):
            num_trap, profit_width, order_size, strategy = params[:4]
            density = params[4] if len(params) > 4 else default_density
            effective_margin, _, _, _, _, _, _, _, _, _ = traripi_backtest(
                calculator, X_train, initial_funds, grid_start, grid_end, num_trap, profit_width, order_size,
                entry_interval, total_threshold, strategy, density
            )
            return -effective_margin

        res = gp_minimize(
            wrapped_objective_function,  # 目的関数
            space,  # 変数の範囲
            n_calls=30,  # 評価の回数
            random_state=0  # 再現性のための乱数シード
        )

        # 最適パラメータでX_testを評価
        best_params = res.x
        best_num_trap = best_params[0]
        best_profit_width = best_params[1]
        best_order_size = best_params[2]
        best_strategy = best_params[3]
        best_density = best_params[4] if len(best_params) > 4 else default_density

        effective_margin, _, _, _, _, _, _, _, _, _ = traripi_backtest(
            calculator, X_test, initial_funds, grid_start, grid_end, best_num_trap, best_profit_width, best_order_size,
            entry_interval, total_threshold, best_strategy, best_density
        )
        margins.append(effective_margin)

    return -np.mean(margins)

# ベイズ最適化の実行
result = gp_minimize(
    func=lambda params: cross_validate_and_optimize(params, space),  # 最適化する関数
    dimensions=space,  # 検索空間
    acq_func='EI',  # 獲得関数 (Expected Improvement)
    n_calls=30,  # 試行回数
    random_state=42  # 再現性のための乱数シード
)

# 結果の表示
print("Best parameters:")
print("num_trap:", result.x[0])
print("profit_width:", result.x[1])
print("order_size:", result.x[2])
print("strategy:", result.x[3])
print("density:", result.x[4])
print("Best score:", -result.fun)
