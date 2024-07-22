import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from mpl_toolkits.mplot3d import Axes3D
from skopt.utils import use_named_args
from kabu_backtest import traripi_backtest, data, initial_funds, grid_start, grid_end, entry_interval, total_threshold
# 最大化したい目的関数（ここでは負の値を返す）
#def objective_function(x):
#    x1, x2 = x
#    # 例として 2次元の関数 (x1^2 - 10*x1 + 25) + (x2^2 - 5*x2 + 6) の負の値を返す
#    return -((x1 ** 2 - 10 * x1 + 25) + (x2 ** 2 - 5 * x2 + 6))

def objective_function(data, initial_funds, grid_start, grid_end, num_traps, profit_width, order_size, entry_interval, total_threshold,strategy,density):
  effective_margin, _, _, _, _, _, _, _, _, _ = traripi_backtest(
                data, initial_funds, grid_start, grid_end, num_traps, profit_width, order_size, entry_interval, total_threshold, strategy, density
            )
  return -effective_margin

# パラメータ空間の定義
space = [
    Integer(1, 101, name='num_traps'),
    Real(0.1, 100, name='profit_width'),
    Integer(1000, 10000, name='order_size'),
    Categorical(['long_only', 'short_only','half_and_half','diamond'], name='strategy'),
    Real(0.1, 10, name='density')
]


# 目的関数をラップする関数
@use_named_args(space)
def wrapped_objective_function(**params):
    return objective_function(
        data, initial_funds, grid_start, grid_end,
        params['num_traps'], params['profit_width'], params['order_size'], 
        entry_interval, total_threshold, params['strategy'], params['density']
    )
	

# ベイズ最適化の実行
result = gp_minimize(
    wrapped_objective_function,                # 目的関数
    space,  # 変数の範囲
    n_calls=500,                        # 評価の回数
    random_state=0                     # 再現性のための乱数シード
)

# 結果の表示
print("最適化されたパラメータ:", result.x)
print("最大化された目的関数の値:", -result.fun)  # 最大値を表示

# 目的関数の最適化過程を可視化（単純な可視化の例）
"""
# 2次元プロットの準備
x1_range = np.linspace(-10, 10, 100)
x2_range = np.linspace(-10, 10, 100)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
z = np.array([objective_function([x1, x2]) for x1, x2 in zip(x1_grid.ravel(), x2_grid.ravel())])
z = z.reshape(x1_grid.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_grid, x2_grid, -z, cmap='viridis')  # 目的関数の最大値を可視化するために負の値を反転

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Objective Function Value')
ax.set_title('3D Surface Plot of the Objective Function (Maximization)')

plt.show()
"""
