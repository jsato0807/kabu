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
    Integer(4, 101, name='num_trap'),
    Real(0.1, 100, name='profit_width'),
    Categorical([1000,2000,3000,4000,5000,6000,7000,8000,9000, 10000], name='order_size'),
    Categorical(['long_only', 'short_only', 'half_and_half','diamond'], name='strategy'),
    Real(1.0,10, name='density')
]

# 固定値
default_density = 1.0

entry_interval = entry_intervals[0]
total_threshold = total_thresholds[0]


# 交差検証を含む目的関数の定義
def objective_function(X_train_cv):
    @use_named_args(space)
    def inner_objective_function(num_trap, profit_width, order_size, strategy, density):
        # トレードバックテストの実行
        effective_margin, _, _, _, _, _, _, _, _, _ = traripi_backtest(
            calculator, X_train_cv, initial_funds, grid_start, grid_end, num_trap, profit_width, order_size,
            entry_interval, total_threshold, strategy, density
        )
        return -effective_margin  # 最大化を最小化として処理する

    # ベイズ最適化の実行
    result = gp_minimize(
        func=inner_objective_function,
        dimensions=space,
        n_calls=50,  # 試行回数の設定
        random_state=42  # 再現性のための乱数シード
    )
    
    # 最適なパラメータでテスト
    best_params = result.x
    best_effective_margin = traripi_backtest(
        calculator, X_test, initial_funds, grid_start, grid_end, *best_params
    )[0]
    
    return best_effective_margin, best_params

# データの準備
kf = KFold(n_splits=10)  # 5-fold クロスバリデーション
results = []

for fold, (train_index, test_index) in enumerate(kf.split(data)):
    X_train_cv, X_test = data[train_index], data[test_index]
    
    # 交差検証内でベイズ最適化を実行
    margin, params = objective_function(X_train_cv)
    results.append({
        'fold': fold + 1,
        'margin': margin,
        'params': params
    })

# 結果の表示
for result in results:
    print(f"Fold {result['fold']}:")
    print(f"  マージン: {result['margin']}")
    print(f"  パラメータ:")
    for param_name, param_value in zip(space, result['params']):
        print(f"    {param_name}: {param_value}")
