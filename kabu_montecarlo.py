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
num_simulations = 1000  # シミュレーションの回数

# 変数として扱うパラメータの範囲
param_ranges = {
    'num_trap': (4, 101),  # 例: 5から20の範囲
    'profit_width': (0.001, 50.0),  # 例: 0.001から0.1の範囲
    'order_size': (1, 10),  # 例: 1から10の範囲
    'strategy': ['diamond', 'long_only', 'short_only', 'half_and_half'],  # 戦略のリスト
    'density': (0.1, 10.0)  # 例: 0.1から10の範囲
}

# パラメータをランダムにサンプリングする関数
def sample_parameters(param_ranges):
    params = {}
    for param, value in param_ranges.items():
        if isinstance(value, tuple):  # 範囲がタプルの場合
            low, high = value
            if isinstance(low, int) and isinstance(high, int):  # 整数の場合
                params[param] = np.random.randint(low, high + 1)
            elif isinstance(low, (int, float)) and isinstance(high, (int, float)):  # 浮動小数点数の場合
                params[param] = np.random.uniform(low, high)
        elif isinstance(value, list):  # リストの場合
            params[param] = np.random.choice(value)
    return params


# 各ペアに対して正規分布のフィッティングとプロット
def fit_and_plot_normal_distribution(df, column_name):
    # 列を選択して NaN を除去
    data = df[column_name].dropna()
    
    # pandas Series を numpy array に変換
    data = data.to_numpy()
    
    # データ型を確認
    if not np.issubdtype(data.dtype, np.number):
        print(f"Column {column_name} contains non-numeric data.")
        return
    data = data[np.isfinite(data)]  # 無限大を除去

    # フィッティング
    mu, std = stats.norm.fit(data)

    # ヒストグラムとフィッティングされた分布のプロット
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='g')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.title(f'Histogram and Fitted Normal Distribution for {column_name}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.savefig(f'{column_name}_histogram.png')  # 各パラメータのヒストグラムを保存
    plt.close()  # プロットを閉じる

    # Q-Qプロット
    plt.figure(figsize=(10, 6))
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(f'Q-Q Plot for {column_name}')
    plt.savefig(f'./png_dir/kabu_montecarlo_{column_name}_qq_plot.png')  # 各パラメータのQ-Qプロットを保存
    plt.close()  # プロットを閉じる


if __name__ == "__main__":
    interval="1d"
    end_date = datetime.strptime("2022-01-01","%Y-%m-%d")#datetime.now() - timedelta(days=7)
    start_date = datetime.strptime("2021-09-01","%Y-%m-%d")#datetime.now() - timedelta(days=14)
    data = data = fetch_currency_data(pair, start_date, end_date,interval)
    # シミュレーション結果を格納するリスト

    results = []
    # モンテカルロシミュレーションの実行
    for _ in range(num_simulations):
        params = sample_parameters(param_ranges)
        effective_margin, _, _, _, _, _, _, _, _, _ = traripi_backtest(
            calculator, data, initial_funds,
            grid_start, grid_end, params['num_trap'],
            params['profit_width'], params['order_size']*1000, entry_interval,
            total_threshold, params['strategy'], params['density']
        )
        params['effective_margin'] = effective_margin
        results.append(params)

    # DataFrame に変換
    df = pd.DataFrame(results)

    # 結果の確認
    print(df.head())
    
    # 各数値パラメータに対してフィッティングとプロットを実施
    for column in param_ranges.keys():
        fit_and_plot_normal_distribution(df, column)

    # ペアプロットの作成
    sns.pairplot(df)
    plt.savefig('./png_dir/kabu_montecarlo_pairplot.png')
    plt.show()