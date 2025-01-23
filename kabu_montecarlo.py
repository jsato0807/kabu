import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import pandas as pd
from datetime import datetime
from kabu_swap import get_html, parse_swap_points, rename_swap_points, SwapCalculator
from kabu_backtest import traripi_backtest, fetch_currency_data, pair, initial_funds, grid_start, grid_end, entry_intervals, total_thresholds, data

entry_interval = entry_intervals
total_threshold = total_thresholds

url = 'https://fx.minkabu.jp/hikaku/moneysquare/spreadswap.html'
html = get_html(url)
swap_points = parse_swap_points(html)
swap_points = rename_swap_points(swap_points)
calculator = SwapCalculator(swap_points)

# モンテカルロシミュレーションのパラメータ設定
num_simulations = 10000  # シミュレーションの回数

# 変数として扱うパラメータの範囲
param_ranges = {
    'num_trap': (4, 101),
    'profit_width': (0.001, 50.0),
    'order_size': (1, 10),
    'strategy': ['diamond', 'long_only', 'short_only', 'half_and_half'],
    'density': (0.1, 10.0)
}

# パラメータをランダムにサンプリングする関数
def sample_parameters(param_ranges):
    params = {}
    for param, value in param_ranges.items():
        if isinstance(value, tuple):
            low, high = value
            if isinstance(low, int) and isinstance(high, int):
                params[param] = np.random.randint(low, high + 1)
            elif isinstance(low, (int, float)) and isinstance(high, (int, float)):
                params[param] = np.random.uniform(low, high)
        elif isinstance(value, list):
            params[param] = np.random.choice(value)
    return params

# ラプラス分布のフィッティングとプロット
def fit_and_plot_laplace_distribution(data, column_name, strategy):
    data = data.dropna()
    data = data.to_numpy()
    if not np.issubdtype(data.dtype, np.number):
        print(f"Column {column_name} contains non-numeric data.")
        return
    data = data[np.isfinite(data)]

    mu, b = stats.laplace.fit(data)

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='g', edgecolor='k')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.laplace.pdf(x, mu, b)
    plt.plot(x, p, 'k', linewidth=2)
    plt.title(f'{strategy} Strategy - Histogram and Fitted Laplace Distribution for {column_name}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.savefig(f'./png_dir/kabu_montecarlo_{strategy}_{column_name}_laplace_histogram.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    stats.probplot(data, dist="laplace", sparams=(mu, b), plot=plt)
    plt.title(f'{strategy} Strategy - Q-Q Plot for {column_name} with Laplace Distribution')
    plt.savefig(f'./png_dir/kabu_montecarlo_{strategy}_{column_name}_laplace_qq_plot.png')
    plt.close()

# 正規分布のフィッティングとプロット
def fit_and_plot_normal_distribution(data, column_name, strategy):
    data = data.dropna()
    data = data.to_numpy()
    if not np.issubdtype(data.dtype, np.number):
        print(f"Column {column_name} contains non-numeric data.")
        return
    data = data[np.isfinite(data)]

    mu, std = stats.norm.fit(data)

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='g', edgecolor='k')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.title(f'{strategy} Strategy - Histogram and Fitted Normal Distribution for {column_name}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.savefig(f'./png_dir/kabu_montecarlo_{strategy}_{column_name}_normal_histogram.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(f'{strategy} Strategy - Q-Q Plot for {column_name} with Normal Distribution')
    plt.savefig(f'./png_dir/kabu_montecarlo_{strategy}_{column_name}_normal_qq_plot.png')
    plt.close()

# 各戦略ごとにペアプロットを作成
def plot_strategy_pairplots(df):
    strategies = df['strategy'].unique()
    for strategy in strategies:
        strategy_df = df[df['strategy'] == strategy]
        plt.figure(figsize=(10, 8))
        sns.pairplot(strategy_df, hue='strategy', plot_kws={'alpha': 0.6})
        plt.suptitle(f'Pairplot for {strategy} Strategy', y=1.02)
        plt.savefig(f'./png_dir/kabu_montecarlo_{strategy}_pairplot.png')
        plt.close()

if __name__ == "__main__":
    interval = "1d"
    end_date = datetime.strptime("2022-01-01", "%Y-%m-%d")
    start_date = datetime.strptime("2019-01-01", "%Y-%m-%d")
    data = fetch_currency_data(pair, start_date, end_date, interval)
    results = []
    for _ in range(num_simulations):
        params = sample_parameters(param_ranges)
        _, _, realized_profit, _, _, _, _, _, _, _ = traripi_backtest(
            calculator, data, initial_funds,
            grid_start, grid_end, params['num_trap'],
            params['profit_width'], params['order_size']*1000, entry_interval,
            total_threshold, params['strategy'], params['density']
        )
        params['realized_profit'] = realized_profit
        results.append(params)

    df = pd.DataFrame(results)
    print(df.head())

    # 各戦略ごとにフィッティングとプロットを実施
    for strategy in param_ranges['strategy']:
        strategy_df = df[df['strategy'] == strategy]
        for column in strategy_df.columns:
            if column in ['num_trap', 'profit_width', 'order_size', 'density', 'realized_profit']:
                fit_and_plot_laplace_distribution(strategy_df[column], column, strategy)
                fit_and_plot_normal_distribution(strategy_df[column], column, strategy)

    # 各戦略ごとにペアプロットを作成
    plot_strategy_pairplots(df)
