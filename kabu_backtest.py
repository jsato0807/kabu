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
        last_price = None  # Variable to store the last processed price

        for i in range(len(data)):
            date = data.index[i]
            price = data.iloc[i]

            # Update effective margin for currently held positions based on current price
            if positions and not pos[2].endswith('Closed'):
                for pos in positions:
                    effective_margin -= pos[0] *(price - pos[3]) - (last_price - pos[3]) # Adjust effective margin for unrealized P/L
            
            if last_price is not None:
                # Check if price has crossed any grid between last_price and price
                if price > last_price:
                    for grid in grids:
                        if last_price <= grid < price:
                            if effective_margin >= order_size * price:
                                effective_margin -= order_size * grid
                                positions.append([order_size, i, 'Buy', grid])
                                trades.append((date, price, 'Buy'))
                                print(f"Opened Buy position at {price} with grid {grid}, Effective Margin: {effective_margin}")
                                break  # Exit loop once position is taken
                elif price < last_price:
                    for grid in grids:
                        if last_price >= grid > price:
                            if effective_margin >= order_size * price:
                                effective_margin -= order_size * grid
                                positions.append([order_size, i, 'Buy', grid])
                                trades.append((date, price, 'Buy'))
                                print(f"Opened Buy position at {price} with grid {grid}, Effective Margin: {effective_margin}")
                                break  # Exit loop once position is taken

            # Update last_price for the next iteration
            last_price = price

          # Position closure processing
            for pos in positions[:]:
                if pos[2] == 'Buy' and pos[1] < len(data) - 1:
                    for future_index in range(pos[1], len(data)):
                        future_price = data.iloc[future_index]
                        if future_price - pos[3] >=  profit_width:
                            effective_margin += order_size * (pos[3] + profit_width*(abs(future_price-pos[3])//profit_width))
                            profit = order_size * profit_width*(abs(future_price-pos[3])//profit_width)
                            realized_profit += profit
                            pos[2] = 'Sell-Closed'
                            trades.append((date, future_price, 'Sell'))
                            print(f"Closed Sell position at {pos[3]+profit_width} with profit {profit} ,grid {pos[3]}, Effective Margin: {effective_margin}")
                            break

    elif strategy == 'short_only':
        grids = np.linspace(grid_start, grid_end, num=num_traps)
        last_price = None  # Variable to store the last processed price

        for i in range(len(data)):
            date = data.index[i]
            price = data.iloc[i]

            # Update effective margin for currently held positions based on current price
            if positions and not pos[2].endswith('Closed'):
                for pos in positions:
                    effective_margin -= pos[0] * ((price) - pos[3]) - (last_price - pos[3])   # Adjust effective margin for unrealized P/L

            if last_price is not None:
                # Check if price has crossed any grid between last_price and price
                if price < last_price:
                    for grid in grids:
                        if last_price >= grid > price:
                            if effective_margin >= order_size * price:
                                effective_margin -= order_size * grid
                                positions.append([order_size, i, 'Sell', grid])
                                trades.append((date, price, 'Sell'))
                                print(f"Opened Sell position at {price} with grid {grid}, Effective Margin: {effective_margin}")
                                break  # Exit loop once position is taken
                elif price > last_price:
                    for grid in grids:
                        if last_price <= grid < price:
                            if effective_margin >= order_size * price:
                                effective_margin -= order_size * grid
                                positions.append([order_size, i, 'Sell', grid])
                                trades.append((date, price, 'Sell'))
                                print(f"Opened Sell position at {price} with grid {grid}, Effective Margin: {effective_margin}")
                                break  # Exit loop once position is taken

            # Update last_price for the next iteration
            last_price = price

            # Position closure processing
            for pos in positions[:]:
                if pos[2] == 'Sell' and pos[1] < len(data) - 1:
                    for future_index in range(pos[1], len(data)):
                        future_price = data.iloc[future_index]
                        if future_price - pos[3] <=  - profit_width:
                            effective_margin += order_size * (pos[3] + profit_width*(abs(future_price-pos[3])//profit_width))
                            profit = order_size * profit_width*(abs(future_price-pos[3])//profit_width)
                            realized_profit += profit
                            pos[2] = 'Buy-Closed'
                            trades.append((date, future_price, 'Buy'))
                            print(f"Closed Buy position at {pos[3]+profit_width} with profit {profit} ,grid {pos[3]}, Effective Margin: {effective_margin}")
                            break

    elif strategy == 'half_and_half':
        half_point = (grid_start + grid_end) / 2
        grids_bottom = np.linspace(grid_start, half_point, num=int(num_traps / 2))
        grids_top = np.linspace(half_point, grid_end, num=int(num_traps / 2))
        last_price = None  # Variable to store the last processed price

        for i in range(len(data)):
            date = data.index[i]
            price = data.iloc[i]

            # Update effective margin for currently held positions based on current price
            if positions and not pos[2].endswith('Closed'):
                for pos in positions:
                    effective_margin -= pos[0] *(price - pos[3]) - (last_price - pos[3]) # Adjust effective margin for unrealized P/L

            if last_price is not None:
                # Check bottom half area
                if price <= half_point:
                    for grid in grids_bottom:
                        if last_price > grid >= price:
                            if effective_margin >= order_size * price:
                                effective_margin -= order_size * grid
                                positions.append([order_size, i, 'Buy', grid])
                                trades.append((date, price, 'Buy'))
                                print(f"Opened Buy position at {price} with grid {grid}, Effective Margin: {effective_margin}")
                                break

                # Check top half area
                if price > half_point:
                    for grid in grids_top:
                        if last_price < grid <= price:
                            if effective_margin >= order_size * price:
                                effective_margin -= order_size * grid
                                positions.append([order_size, i, 'Sell', grid])
                                trades.append((date, price, 'Sell'))
                                print(f"Opened Sell position at {price} with grid {grid}, Effective Margin: {effective_margin}")
                                break

            # Update last_price for the next iteration
            last_price = price

            # Position closure processing
            for pos in positions[:]:
                if pos[2] == 'Buy' and pos[1] < len(data) - 1:
                    for future_index in range(pos[1], len(data)):
                        future_price = data.iloc[future_index]
                        if future_price - pos[3] >=  profit_width:
                            effective_margin += order_size * (pos[3] + profit_width*(abs(future_price-pos[3])//profit_width))
                            profit = order_size * profit_width*(abs(future_price-pos[3])//profit_width)
                            realized_profit += profit
                            pos[2] = 'Sell-Closed'
                            trades.append((date, future_price, 'Sell'))
                            print(f"Closed Sell position at {pos[3]+profit_width} with profit {profit} ,grid {pos[3]}, Effective Margin: {effective_margin}")
                            break
                elif pos[2] == 'Sell' and pos[1] < len(data) - 1:
                    for future_index in range(pos[1], len(data)):
                        future_price = data.iloc[future_index]
                        if future_price - pos[3] <=  - profit_width:
                            effective_margin += order_size * (pos[3] + profit_width*(abs(future_price-pos[3])//profit_width))
                            profit = order_size * profit_width*(abs(future_price-pos[3])//profit_width)
                            realized_profit += profit
                            pos[2] = 'Buy-Closed'
                            trades.append((date, future_price, 'Buy'))
                            print(f"Closed Buy position at {pos[3]+profit_width} with profit {profit} ,grid {pos[3]}, Effective Margin: {effective_margin}")
                            break

    elif strategy == 'diamond':
        quarter_point = (grid_start + grid_end) / 4
        half_point = (grid_start + grid_end) / 2
        three_quarter_point = 3 * (grid_start + grid_end) / 4

        # Set grid numbers for each area
        grids_bottom = np.linspace(grid_start, quarter_point, num=int(num_traps / 2))
        grids_lower_center = np.linspace(quarter_point, half_point, num=int(num_traps * density))
        grids_upper_center = np.linspace(half_point, three_quarter_point, num=int(num_traps * density))
        grids_top = np.linspace(three_quarter_point, grid_end, num=int(num_traps / 2))
        last_price = None  # Variable to store the last processed price

        for i in range(len(data)):
            date = data.index[i]
            price = data.iloc[i]

            # Update effective margin for currently held positions based on current price
            if positions and not pos[2].endswith('Closed'):
                for pos in positions:
                    effective_margin -= pos[0] * (price - pos[3]) - (last_price - pos[3])  # Adjust effective margin for unrealized P/L

            if last_price is not None:
                # Check bottom two areas
                if price <= half_point:
                    for grid in np.concatenate([grids_bottom, grids_lower_center]):
                        if last_price > grid >= price:
                            if effective_margin >= order_size * price:
                                effective_margin -= order_size * grid
                                positions.append([order_size, i, 'Buy', grid])
                                trades.append((date, price, 'Buy'))
                                print(f"Opened Buy position at {price} with grid {grid}, Effective Margin: {effective_margin}")
                                break

                # Check top two areas
                if price >= half_point:
                    for grid in np.concatenate([grids_top, grids_upper_center]):
                        if last_price < grid <= price:
                            if effective_margin >= order_size * price:
                                effective_margin -= order_size * grid
                                positions.append([order_size, i, 'Sell', grid])
                                trades.append((date, price, 'Sell'))
                                print(f"Opened Sell position at {price} with grid {grid}, Effective Margin: {effective_margin}")
                                break

            # Update last_price for the next iteration
            last_price = price

            # Position closure processing
            for pos in positions[:]:
                if pos[2] == 'Buy' and pos[1] < len(data) - 1:
                    for future_index in range(pos[1], len(data)):
                      if future_price - pos[3] >=  profit_width:
                          future_price = data.iloc[future_index]
                          effective_margin += order_size * (pos[3] + profit_width*(abs(future_price-pos[3])//profit_width))
                          profit = order_size * profit_width*(abs(future_price-pos[3])//profit_width)
                          realized_profit += profit
                          pos[2] = 'Sell-Closed'
                          trades.append((date, future_price, 'Sell'))
                          print(f"Closed Sell position at {future_price} with profit {profit} ,grid {pos[3]}, Effective Margin: {effective_margin}")
                          break
                elif pos[2] == 'Sell' and pos[1] < len(data) - 1:
                    for future_index in range(pos[1], len(data)):  
                        future_price = data.iloc[future_index]
                        if future_price - pos[3] <=  - profit_width:
                            effective_margin += order_size * (pos[3] + profit_width*(abs(future_price-pos[3])//profit_width))
                            profit = order_size * profit_width*(abs(future_price-pos[3])//profit_width)
                            realized_profit += profit
                            pos[2] = 'Buy-Closed'
                            trades.append((date, future_price, 'Buy'))
                            print(f"Closed Buy position at {future_price} with profit {profit} ,grid {pos[3]}, Effective Margin: {effective_margin}")
                            break

    # Calculate position value
    if positions:
        position_value = sum(size * (data.iloc[-1] - grid) if 'Buy' in status and not status.endswith('Closed') else
                     -size * (data.iloc[-1] - grid) if 'Sell'  in status and not status.endswith('Closed') else
                     0 for size, _, status, grid in positions)
    else:
        position_value = 0

    # Calculate margin deposit
    margin_deposit = initial_funds + realized_profit

    return effective_margin, margin_deposit, realized_profit, position_value, trades

# パラメータ設定
pair = "USDJPY=X"
start_date = "2020-01-01"
end_date = "2023-01-01"
initial_funds = 1000000
grid_start = 100
grid_end = 110
order_sizes = [1000]
num_traps_options = [2]
profit_widths = [5]
strategies = ['short_only']
densities = [1]

# データの取得
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

# 結果の表示
# ユニークな組み合わせを取得
unique_results = results_df.drop_duplicates(subset=['Order Size', 'Num Traps', 'Profit Width', 'Strategy', 'Density'])

# Top 5 Results Based on Effective Margin
print("上位5件の有効証拠金に基づく結果:")
rank = 1
seen_results = set()  # 重複を管理するためのセット
for i, row in results_df.sort_values(by='Effective Margin', ascending=False).iterrows():
    key = (row['Margin Deposit'], row['Effective Margin'], row['Position Value'], row['Realized Profit'])
    if key in seen_results:
        continue
    seen_results.add(key)
    print(f"Rank {rank}:")
    print(f"  預託証拠金: {row['Margin Deposit']}")
    print(f"  有効証拠金: {row['Effective Margin']}")
    print(f"  評価損益: {row['Position Value']}")
    print(f"  確定利益: {row['Realized Profit']}")
    print(f"  取引通貨量: {row['Order Size']}, トラップ本数: {row['Num Traps']}, 利益値幅: {row['Profit Width']}, 戦略: {row['Strategy']}, 密度: {row['Density']}")
    rank += 1
    if rank > 5:
        break

print("\n最悪の3件の有効証拠金に基づく結果:")
rank = 1
seen_results = set()  # 重複を管理するためのセット
for i, row in results_df.sort_values(by='Effective Margin').head(3).iterrows():
    key = (row['Margin Deposit'], row['Effective Margin'], row['Position Value'], row['Realized Profit'])
    if key in seen_results:
        continue
    seen_results.add(key)
    print(f"Rank {rank}:")
    print(f"  預託証拠金: {row['Margin Deposit']}")
    print(f"  有効証拠金: {row['Effective Margin']}")
    print(f"  評価損益: {row['Position Value']}")
    print(f"  確定利益: {row['Realized Profit']}")
    print(f"  取引通貨量: {row['Order Size']}, トラップ本数: {row['Num Traps']}, 利益値幅: {row['Profit Width']}, 戦略: {row['Strategy']}, 密度: {row['Density']}")
    rank += 1
