import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from itertools import product

def fetch_currency_data(pair, start, end, interval='1d'):
    """
    Fetch historical currency pair data from Yahoo Finance.
    """
    data = yf.download(pair, start=start, end=end, interval=interval)
    data = data['Close']
    print(f"Fetched data length: {len(data)}")
    return data

def traripi_backtest(data, initial_funds, grid_start, grid_end, num_traps, profit_width, order_size, strategy='standard', density=1):
    """
    Perform Trailing Stop strategy backtest on given data.
    """
    effective_margin = initial_funds
    realized_profit = 0
    positions = []
    trades = []

    if strategy == 'long_only':
        grids = np.linspace(grid_start, grid_end, num=num_traps)

        for i in range(len(data)):
            date = data.index[i]
            price = data.iloc[i]

            match_counts = np.zeros(num_traps)

            for j, grid in enumerate(grids):
                if price <= grid and effective_margin >= order_size * price:
                    match_counts[j] += 1

            best_match_index = np.argmax(match_counts)

            if match_counts[best_match_index] > 0:
                effective_margin -= order_size * price
                positions.append((order_size, price, 'Buy'))
                trades.append((date, price, 'Buy'))

            # Position closure processing
            for pos in positions[:]:
                if pos[2] == 'Buy' and i < len(data) - 1:
                    future_price = data.iloc[i + 1]
                    if future_price >= pos[1] + profit_width:
                        effective_margin += order_size * future_price
                        profit = order_size * (future_price - pos[1])
                        realized_profit += profit
                        positions.remove(pos)
                        trades.append((date, future_price, 'Sell'))

    elif strategy == 'short_only':
        grids = np.linspace(grid_start, grid_end, num=num_traps)

        for i in range(len(data)):
            date = data.index[i]
            price = data.iloc[i]

            match_counts = np.zeros(num_traps)

            for j, grid in enumerate(grids):
                if price >= grid and effective_margin >= order_size * price:
                    match_counts[j] += 1

            best_match_index = np.argmax(match_counts)

            if match_counts[best_match_index] > 0:
                effective_margin -= order_size * price
                positions.append((order_size, price, 'Sell'))
                trades.append((date, price, 'Sell'))

            # Position closure processing
            for pos in positions[:]:
                if pos[2] == 'Sell' and i < len(data) - 1:
                    future_price = data.iloc[i + 1]
                    if future_price <= pos[1] - profit_width:
                        effective_margin += order_size * future_price
                        profit = order_size * (pos[1] - future_price)
                        realized_profit += profit
                        positions.remove(pos)
                        trades.append((date, future_price, 'Buy'))

    elif strategy == 'half_and_half':
        half_point = (grid_start + grid_end) / 2
        grids_bottom = np.linspace(grid_start, half_point, num=int(num_traps / 2))
        grids_top = np.linspace(half_point, grid_end, num=int(num_traps / 2))

        for i in range(len(data)):
            date = data.index[i]
            price = data.iloc[i]

            # Bottom half area trap setting (enter from buy)
            if price <= half_point and effective_margin >= order_size * price:
                match_counts_bottom = np.zeros(int(num_traps / 2))

                for j, grid in enumerate(grids_bottom):
                    if price >= grid and effective_margin >= order_size * price:
                        match_counts_bottom[j] += 1

                best_match_index_bottom = np.argmax(match_counts_bottom)

                if match_counts_bottom[best_match_index_bottom] > 0:
                    effective_margin -= order_size * price
                    positions.append((order_size, price, 'Buy'))
                    trades.append((date, price, 'Buy'))

            # Top half area trap setting (enter from sell)
            if price >= half_point and effective_margin >= order_size * price:
                match_counts_top = np.zeros(int(num_traps / 2))

                for j, grid in enumerate(grids_top):
                    if price <= grid and effective_margin >= order_size * price:
                        match_counts_top[j] += 1

                best_match_index_top = np.argmax(match_counts_top)

                if match_counts_top[best_match_index_top] > 0:
                    effective_margin -= order_size * price
                    positions.append((order_size, price, 'Sell'))
                    trades.append((date, price, 'Sell'))

            # Position closure processing
            for pos in positions[:]:
                if pos[2] == 'Buy' and i < len(data) - 1:
                    future_price = data.iloc[i + 1]
                    if future_price >= pos[1] + profit_width:
                        effective_margin += order_size * future_price
                        profit = order_size * (future_price - pos[1])
                        realized_profit += profit
                        positions.remove(pos)
                        trades.append((date, future_price, 'Sell'))
                elif pos[2] == 'Sell' and i < len(data) - 1:
                    future_price = data.iloc[i + 1]
                    if future_price <= pos[1] - profit_width:
                        effective_margin += order_size * future_price
                        profit = order_size * (pos[1] - future_price)
                        realized_profit += profit
                        positions.remove(pos)
                        trades.append((date, future_price, 'Buy'))

    elif strategy == 'diamond':
        quarter_point = (grid_start + grid_end) / 4
        half_point = (grid_start + grid_end) / 2
        three_quarter_point = 3 * (grid_start + grid_end) / 4

        # Set grid numbers for each area
        grids_bottom = np.linspace(grid_start, quarter_point, num=int(num_traps / 2))
        grids_lower_center = np.linspace(quarter_point, half_point, num=int(num_traps * density))
        grids_upper_center = np.linspace(half_point, three_quarter_point, num=int(num_traps * density))
        grids_top = np.linspace(three_quarter_point, grid_end, num=int(num_traps / 2))

        for i in range(len(data)):
            date = data.index[i]
            price = data.iloc[i]

            # Bottom two areas trap setting (enter from buy)
            if price <= half_point and effective_margin >= order_size * price:
                for grid in np.concatenate([grids_bottom, grids_lower_center]):
                    if price <= grid and effective_margin >= order_size * price:
                        effective_margin -= order_size * price
                        positions.append((order_size, price, 'Buy'))
                        trades.append((date, price, 'Buy'))
                        break

            # Top two areas trap setting (enter from sell)
            if price >= half_point and effective_margin >= order_size * price:
                for grid in np.concatenate([grids_top, grids_upper_center]):
                    if price >= grid and effective_margin >= order_size * price:
                        effective_margin -= order_size * price
                        positions.append((order_size, price, 'Sell'))
                        trades.append((date, price, 'Sell'))
                        break

            # Position closure processing
            for pos in positions[:]:
                if pos[2] == 'Buy' and i < len(data) - 1:
                    future_price = data.iloc[i + 1]
                    if future_price >= pos[1] + profit_width:
                        effective_margin += order_size * future_price
                        profit = order_size * (future_price - pos[1])
                        realized_profit += profit
                        positions.remove(pos)
                        trades.append((date, future_price, 'Sell'))
                elif pos[2] == 'Sell' and i < len(data) - 1:
                    future_price = data.iloc[i + 1]
                    if future_price <= pos[1] - profit_width:
                        effective_margin += order_size * future_price
                        profit = order_size * (pos[1] - future_price)
                        realized_profit += profit
                        positions.remove(pos)
                        trades.append((date, future_price, 'Buy'))

    # Calculate position value
    if positions:
        position_value = sum(size * (data.iloc[-1] - price) for size, price, _ in positions)
    else:
        position_value = 0

    total_value = effective_margin + realized_profit

    return effective_margin, total_value, realized_profit, position_value, trades

# Parameter setup
pair = "USDJPY=X"
start_date = "2020-01-01"
end_date = "2023-01-01"
initial_funds = 1000000
grid_start = 100
grid_end = 150
order_sizes = [1000,2000]
num_traps_options = [50,100]
profit_widths = [1,2]
#strategies = ['half_and_half','long_only','short_only','diamond']
strategies = ['long_only','short_only']
densities = [1,2]

# Fetch data
data = fetch_currency_data(pair, start=start_date, end=end_date)

results = []

# バックテストの実行
for order_size, num_traps, profit_width, strategy, density in product(order_sizes, num_traps_options, profit_widths, strategies, densities):
    effective_margin, margin_deposit, realized_profit, position_value, trades = traripi_backtest(
        data, initial_funds, grid_start, grid_end, num_traps, profit_width, order_size, strategy=strategy, density=density
    )
    results.append((effective_margin, margin_deposit, realized_profit, position_value, order_size, num_traps, profit_width, strategy, density))

# 結果の表示
results_df = pd.DataFrame(results, columns=[
    'Effective Margin', 'Margin Deposit', 'Realized Profit', 'Position Value', 'Order Size', 'Num Traps', 'Profit Width', 'Strategy', 'Density'
])

# 有効証拠金でソートして上位と下位の結果を抽出
sorted_results = results_df.sort_values(by='Effective Margin', ascending=False)
top_results = sorted_results.head(5)
worst_results = sorted_results.tail(3)

# トップ5の結果を表示
print("Top 5 Results Based on Effective Margin:")
for i, row in top_results.iterrows():
    print(f"Rank {i+1}:")
    print(f"  預託証拠金: {row['Margin Deposit']}")
    print(f"  有効証拠金: {row['Effective Margin']}")
    print(f"  評価損益: {row['Position Value']}")
    print(f"  確定利益: {row['Realized Profit']}")
    print(f"  取引通貨量: {row['Order Size']}, トラップ本数: {row['Num Traps']}, 利益値幅: {row['Profit Width']}, 戦略: {row['Strategy']}, 密度: {row['Density']}")

# ワースト3の結果を表示
print("\nWorst 3 Results Based on Effective Margin:")
for i, row in worst_results.iterrows():
    print(f"Rank {len(sorted_results) - i}:")
    print(f"  預託証拠金: {row['Margin Deposit']}")
    print(f"  有効証拠金: {row['Effective Margin']}")
    print(f"  評価損益: {row['Position Value']}")
    print(f"  確定利益: {row['Realized Profit']}")
    print(f"  取引通貨量: {row['Order Size']}, トラップ本数: {row['Num Traps']}, 利益値幅: {row['Profit Width']}, 戦略: {row['Strategy']}, 密度: {row['Density']}")
