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
    margin_deposit = initial_funds
    effective_margin = margin_deposit
    realized_profit = 0
    required_margin = 0		# current_rate*order_size*required_margin_rate
    margin_maintenance_rate = float('inf')		# effective_margin/required_margin*100
    required_margin_rate = 0.04
    positions = []
    trades = []
    margin_maintenance_flag = False
    order_capacity_flag = False

    if strategy == 'long_only':
        grids = np.linspace(grid_start, grid_end, num=num_traps)
        order_margin = sum(order_size * grid * required_margin_rate for grid in grids)
        order_capacity = effective_margin - (required_margin + order_margin)
        if order_capacity <= 0:
            print(f'cannot order because of lack of order_capacity')
            order_capacity_flag = True
            
        last_price = None  # Variable to store the last processed price

        for i in range(len(data)):
            if margin_maintenance_flag or order_capacity_flag:
                break
            date = data.index[i]
            price = data.iloc[i]


            if last_price is not None:
                # Check if price has crossed any grid between last_price and price
                if price > last_price:
                    for grid in grids:
                        if last_price <= grid < price:
                            if margin_maintenance_rate <= 100:
                                margin_maintenance_flag = True
                                break
                            order_capacity = effective_margin - (required_margin + order_margin)
                            if order_capacity < 0:
                                order_capacity_flag = True
                                print(f'cannot order because of lack of order capacity')
                                break
                            if margin_maintenance_rate > 100 and order_capacity > 0:		#you dont need to confirm that order_capacity is more than 0 or not because order_margin and required_margin is equal so order_capacity is more than 0 absolutely
                                #margin_deposit -= order_size * grid
                                #effective_margin -= order_size * grid
                                order_margin -= order_size * grid * required_margin_rate
                                add_required_margin = grid * order_size * required_margin_rate
                                required_margin += add_required_margin
                                positions.append([order_size, i, 'Buy', grid, 0, add_required_margin])
                                trades.append((date, price, 'Buy'))
                                print(f"Opened Buy position at {price} with grid {grid}, Effective Margin: {effective_margin}")
                                #break  # Exit loop once position is taken
                                if abs(required_margin) > 0:
                                    margin_maintenance_rate = effective_margin / required_margin * 100
                                    if margin_maintenance_rate <= 100:
                                        print("executed loss cut in last_price <= grid < price")
                                        margin_maintenance_flag = True
                                        break
                                else:
                                    margin_maintenance_rate = float('inf')
                                

                elif price < last_price:
                    for grid in grids:
                        if last_price >= grid > price:
                            if margin_maintenance_rate <= 100:
                                margin_maintenance_flag = True
                                break
                            order_capacity = effective_margin - (required_margin + order_margin)
                            if order_capacity < 0:
                                order_capacity_flag = True
                                print(f'cannot order because of lack of order capacity')
                                break
                            if margin_maintenance_rate > 100 and order_capacity > 0:
                                #margin_deposit -= order_size * grid
                                #effective_margin -= order_size * grid
                                order_margin -= order_size * grid * required_margin_rate
                                add_required_margin = grid * order_size * required_margin_rate
                                required_margin += add_required_margin
                                positions.append([order_size, i, 'Buy', grid, 0, add_required_margin])
                                trades.append((date, price, 'Buy'))
                                print(f"Opened Buy position at {price} with grid {grid}, Effective Margin: {effective_margin}")
                                #break  # Exit loop once position is taken

                                if abs(required_margin) > 0:
                                    margin_maintenance_rate = effective_margin / required_margin * 100
                                    if margin_maintenance_rate <= 100:
                                        print("executed loss cut in last_price >= grid > price")
                                        margin_maintenance_flag = True
                                        break
                                else:
                                    margin_maintenance_rate = float('inf')
                                

          # Position closure processing
            for pos in positions[:]:
                if pos[2] == 'Buy' and pos[1] < len(data) - 1:
                    if price - pos[3] < profit_width:
                        # Update unrealized profit for open positions
                        unrealized_profit = order_size * (price - pos[3])
                        effective_margin += unrealized_profit -  pos[4]  # Adjust for previous unrealized profit
                        add_required_margin = -pos[5] + price * order_size * required_margin_rate
                        required_margin += add_required_margin
                        pos[4] = unrealized_profit  # Store current unrealized profit in the position
                        pos[5] += add_required_margin
                        print(f"updated effective margin against price {price} , Effective Margin: {effective_margin}")

                        if abs(required_margin) > 0:
                            margin_maintenance_rate = effective_margin / required_margin * 100
                            if margin_maintenance_rate <= 100:
                                print("executed loss cut in price - pos[3] < profit_width")
                                margin_maintenance_flag = True
                                continue
                        else:
                            margin_maintenance_rate = float('inf')

                    if price - pos[3] >=  profit_width:
                        effective_margin += order_size * profit_width - pos[4]
                        margin_deposit += order_size * profit_width 
                        profit = order_size * profit_width
                        realized_profit += profit
                        required_margin -= pos[5]
                        pos[5] = 0
                        pos[2] = 'Sell-Closed'
                        trades.append((date, price, 'Sell'))
                        print(f"Closed Sell position at {pos[3]+profit_width} with profit {profit} ,grid {pos[3]}, Effective Margin: {effective_margin}, Required Margin: {required_margin}")
                        #break

                        if abs(required_margin) > 0:
                            margin_maintenance_rate = effective_margin / required_margin * 100
                            if margin_maintenance_rate <= 100:
                                print("executed loss cut in price - pos[3] >=  profit_width")
                                margin_maintenance_flag = True
                                continue
                        else:
                            margin_maintenance_rate = float('inf')

            # Update last_price for the next iteration
            last_price = price


           # 強制ロスカットのチェック
            while positions and margin_maintenance_flag:
                pos = positions.pop(0)
                if pos[2] == 'Buy':
                    profit = (price - pos[3]) * order_size  # 現在の損失計算
                    effective_margin += profit - pos[4] # 損失分を証拠金に反映
                    margin_deposit += profit
                    realized_profit += profit
                    required_margin -= pos[5]
                    pos[5] = 0
                    trades.append((date, price, 'Forced Closed'))
                    print(f"Forced Closed at {price} with grid {pos[3]}, Effective Margin: {effective_margin}")
                    if abs(required_margin) > 0:
                        margin_maintenance_rate = effective_margin / required_margin * 100
                    else:
                        margin_maintenance_rate = float('inf')

    # Calculate position value
    if positions:
        position_value = sum(size * (data.iloc[-1] - grid) if 'Buy' in status and not status.endswith('Closed') else
                     -size * (data.iloc[-1] - grid) if 'Sell'  in status and not status.endswith('Closed') else
                     0 for size, _, status, grid, _, _ in positions)
    else:
        position_value = 0

    ## Calculate margin deposit
    #margin_deposit = initial_funds + realized_profit

    return effective_margin, margin_deposit, realized_profit, position_value, required_margin, margin_maintenance_rate, trades

# パラメータ設定
pair = "USDJPY=X"
start_date = "2020-01-01"
end_date = "2023-01-01"
initial_funds = 1000000
grid_start = 100
grid_end = 110
order_sizes = [1000]
num_traps_options = [1100]
profit_widths = [2]
strategies = ['long_only']
densities = [2]

# データの取得
data = fetch_currency_data(pair, start=start_date, end=end_date)

results = []


# バックテストの実行
for order_size, num_traps, profit_width, strategy, density in product(order_sizes, num_traps_options, profit_widths, strategies, densities):
    effective_margin, margin_deposit, realized_profit, position_value, required_margin, margin_maintenance_rate, trades = traripi_backtest(
        data, initial_funds, grid_start, grid_end, num_traps, profit_width, order_size, strategy=strategy, density=density
    )
    results.append((effective_margin, margin_deposit, realized_profit, position_value, order_size, num_traps, profit_width, strategy, density, required_margin, margin_maintenance_rate))

    

# 結果の表示
results_df = pd.DataFrame(results, columns=[
    'Effective Margin', 'Margin Deposit', 'Realized Profit', 'Position Value', 'Order Size', 'Num Traps', 'Profit Width', 'Strategy', 'Density','Required Margin', 'Margin Maintenance Rate'
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
    print(f" 必要証拠金: {row['Required Margin']}")
    print(f"証拠金維持率: {row['Margin Maintenance Rate']}")
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
    print(f" 必要証拠金: {row['Required Margin']}")
    print(f"証拠金維持率: {row['Margin Maintenance Rate']}")
    print(f"  取引通貨量: {row['Order Size']}, トラップ本数: {row['Num Traps']}, 利益値幅: {row['Profit Width']}, 戦略: {row['Strategy']}, 密度: {row['Density']}")
    rank += 1

""""
if trades:
    dates = [trade[0] for trade in trades]
    prices = [trade[1] for trade in trades]
    actions = [trade[2] for trade in trades]

    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data.values, label=pair)
    plt.scatter(dates, prices, marker='o', c='r' if actions[0] == 'Buy' else 'g', label='Buy' if actions[0] == 'Buy' else 'Sell')
    plt.title(f"{pair} Price and Trades")
    plt.legend()
    plt.show()
"""