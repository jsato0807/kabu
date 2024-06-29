import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from itertools import product

def fetch_currency_data(pair, start, end):
    """
    指定した通貨ペアの過去データを取得する関数
    """
    data = yf.download(pair, start=start, end=end)
    data = data['Close']
    return data

def traripi_backtest(data, total_funds, grid_start, grid_end, num_traps, profit_width, order_size, strategy='standard', density=1):
    """
    トラリピ戦略のバックテストを行う関数
    """
    effective_margin = total_funds
    positions = []
    trades = []
    total_realized_profit = 0  # 確定利益の合計

    if strategy == 'half_and_half':
        half_point = (grid_start + grid_end) / 2
        grids_bottom = np.linspace(grid_start, half_point, num=int(num_traps / 2))
        grids_top = np.linspace(half_point, grid_end, num=int(num_traps / 2))

        for date, price in data.items():
            # 下半分のエリアのトラップ設定（買いから入る）
            if price <= half_point and effective_margin >= order_size * price:
                for grid in np.concatenate([grids_bottom]):
                    if price <= grid and effective_margin >= order_size * price:
                        effective_margin -= order_size * price
                        positions.append((order_size, price))
                        trades.append((date, price, 'Buy'))
                        break

            # 上半分のエリアのトラップ設定（売りから入る）
            if price >= half_point and effective_margin >= order_size * price:
                for grid in np.concatenate([grids_top]):
                    if price >= grid and effective_margin >= order_size * price:
                        effective_margin -= order_size * price
                        positions.append((order_size, price))
                        trades.append((date, price, 'Sell'))
                        break

            # ポジションの決済処理
            for pos in positions[:]:
                if pos[0] == order_size:
                    if pos[1] <= half_point and (price >= half_point + profit_width):
                        effective_margin += order_size * price
                        profit = order_size * (price - pos[1])
                        total_realized_profit += profit
                        positions.remove(pos)
                        trades.append((date, price, 'Sell'))
                    elif pos[1] > half_point and (price <= half_point - profit_width):
                        effective_margin += order_size * price
                        profit = order_size * (pos[1] - price)
                        total_realized_profit += profit
                        positions.remove(pos)
                        trades.append((date, price, 'Buy'))

    elif strategy == 'short_only':
        for date, price in data.items():
            # ショートのみ戦略のトラップ設定（売りから入る）
            if price <= grid_end and effective_margin >= order_size * price:
                effective_margin -= order_size * price
                positions.append((order_size, price))
                trades.append((date, price, 'Sell'))

            # ポジションの決済処理
            for pos in positions[:]:
                if pos[0] == order_size:
                    if price >= pos[1] + profit_width:
                        effective_margin += order_size * price
                        profit = order_size * (pos[1] - price)
                        total_realized_profit += profit
                        positions.remove(pos)
                        trades.append((date, price, 'Buy'))

    elif strategy == 'long_only':
        for date, price in data.items():
            # ロングのみ戦略のトラップ設定（買いから入る）
            if price >= grid_start and effective_margin >= order_size * price:
                effective_margin -= order_size * price
                positions.append((order_size, price))
                trades.append((date, price, 'Buy'))

            # ポジションの決済処理
            for pos in positions[:]:
                if pos[0] == order_size:
                    if price <= pos[1] - profit_width:
                        effective_margin += order_size * price
                        profit = order_size * (price - pos[1])
                        total_realized_profit += profit
                        positions.remove(pos)
                        trades.append((date, price, 'Sell'))

    elif strategy == 'diamond':
        quarter_point = (grid_start + grid_end) / 4
        half_point = (grid_start + grid_end) / 2
        three_quarter_point = 3 * (grid_start + grid_end) / 4

        # 各エリアごとのgrid数を設定
        grids_bottom = np.linspace(grid_start, quarter_point, num=int(num_traps / 2))
        grids_lower_center = np.linspace(quarter_point, half_point, num=int(num_traps * density))
        grids_upper_center = np.linspace(half_point, three_quarter_point, num=int(num_traps * density))
        grids_top = np.linspace(three_quarter_point, grid_end, num=int(num_traps / 2))

        for date, price in data.items():
            # 下から２つのエリアのトラップ設定（買いから入る）
            if price <= half_point and effective_margin >= order_size * price:
                for grid in np.concatenate([grids_bottom, grids_lower_center]):
                    if price <= grid and effective_margin >= order_size * price:
                        effective_margin -= order_size * price
                        positions.append((order_size, price))
                        trades.append((date, price, 'Buy'))
                        break

            # 上から２つのエリアのトラップ設定（売りから入る）
            if price >= half_point and effective_margin >= order_size * price:
                for grid in np.concatenate([grids_top, grids_upper_center]):
                    if price >= grid and effective_margin >= order_size * price:
                        effective_margin -= order_size * price
                        positions.append((order_size, price))
                        trades.append((date, price, 'Sell'))
                        break

            # ポジションの決済処理
            for pos in positions[:]:
                if pos[0] == order_size:
                    if pos[1] <= half_point and (price >= half_point + profit_width):
                        effective_margin += order_size * price
                        profit = order_size * (price - pos[1])
                        total_realized_profit += profit
                        positions.remove(pos)
                        trades.append((date, price, 'Sell'))
                    elif pos[1] > half_point and (price <= half_point - profit_width):
                        effective_margin += order_size * price
                        profit = order_size * (pos[1] - price)
                        total_realized_profit += profit
                        positions.remove(pos)
                        trades.append((date, price, 'Buy'))

    # ポジションの評価損益を計算
    if positions:
        position_value = sum(size * (data.iloc[-1] - price) for size, price in positions)
    else:
        position_value = 0
    
    total_value = effective_margin + total_realized_profit

    return effective_margin, total_value, total_realized_profit, trades

# パラメータの設定
pair = "USDJPY=X"
start_date = "2020-01-01"
end_date = "2023-01-01"
total_funds = 1000000  # 総資金
grid_start = 100  # トラリピ開始レート
grid_end = 150  # トラリピ終了レート

# テストするパラメータの組み合わせ
order_sizes = [1000, 2000, 3000]  # 任意の取引通貨量
num_traps_options = [10, 20, 30]  # トラップ本数
profit_widths = [0.5, 1.0, 1.5]  # 利益値幅
strategies = ['half_and_half', 'short_only', 'long_only', 'diamond']
densities = [1, 2, 3]  # densityの値

# データの取得
data = fetch_currency_data(pair, start=start_date, end=end_date)

results = []

# バックテストの実行
for order_size, num_traps, profit_width, strategy, density in product(order_sizes, num_traps_options, profit_widths, strategies, densities):
    effective_margin, total_value, total_realized_profit, trades = traripi_backtest(
        data, total_funds, grid_start, grid_end, num_traps, profit_width, order_size, strategy=strategy, density=density
    )
    results.append((total_value, effective_margin, total_realized_profit, order_size, num_traps, profit_width, strategy,density))


# Sorting results based on effective margin (descending)
sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

# Displaying top 5 and worst 3 results based on effective margin
top_results = sorted_results[:5]
worst_results = sorted_results[-3:]

# Displaying the top 5 results based on effective margin
print("Top 5 Results Based on Effective Margin:")
for idx, result in enumerate(top_results[:5], 1):
    total_value, effective_margin, total_realized_profit, order_size, num_traps, profit_width, strategy, density = result
    if strategy == 'diamond':
        print(f"Rank {idx}:")
        print(f"  預託証拠金: {total_value:.2f}")
        print(f"  有効証拠金: {effective_margin:.2f}")
        print(f"  評価損益: {total_realized_profit:.2f}")
        print(f"  確定利益: {total_realized_profit:.2f}")
        print(f"  取引通貨量: {order_size}, トラップ本数: {num_traps}, 利益値幅: {profit_width}, 戦略: {strategy}, 密度: {density}")
    else:
        print(f"Rank {idx}:")
        print(f"  預託証拠金: {total_value:.2f}")
        print(f"  有効証拠金: {effective_margin:.2f}")
        print(f"  評価損益: {total_realized_profit:.2f}")
        print(f"  確定利益: {total_realized_profit:.2f}")
        print(f"  取引通貨量: {order_size}, トラップ本数: {num_traps}, 利益値幅: {profit_width}, 戦略: {strategy}")
    print()

# Displaying the worst 3 results based on effective margin
print("Worst 3 Results Based on Effective Margin:")
for idx, result in enumerate(worst_results[-3:], 1):
    total_value, effective_margin, total_realized_profit, order_size, num_traps, profit_width, strategy, density = result
    if strategy == 'diamond':
        print(f"Rank {len(sorted_results) - 3 + idx}:")
        print(f"  預託証拠金: {total_value:.2f}")
        print(f"  有効証拠金: {effective_margin:.2f}")
        print(f"  評価損益: {total_realized_profit:.2f}")
        print(f"  確定利益: {total_realized_profit:.2f}")
        print(f"  取引通貨量: {order_size}, トラップ本数: {num_traps}, 利益値幅: {profit_width}, 戦略: {strategy}, 密度: {density}")
    else:
        print(f"Rank {len(sorted_results) - 3 + idx}:")
        print(f"  預託証拠金: {total_value:.2f}")
        print(f"  有効証拠金: {effective_margin:.2f}")
        print(f"  評価損益: {total_realized_profit:.2f}")
        print(f"  確定利益: {total_realized_profit:.2f}")
        print(f"  取引通貨量: {order_size}, トラップ本数: {num_traps}, 利益値幅: {profit_width}, 戦略: {strategy}")
    print()
