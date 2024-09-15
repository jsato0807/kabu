import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from itertools import product
from kabu_swap import SwapCalculator, get_html, parse_swap_points, add_business_days, rename_swap_points
from datetime import datetime, timedelta

def fetch_currency_data(pair, start, end, interval):
    """
    Fetch historical currency pair data from Yahoo Finance.
    """
    data = yf.download(pair, start=start, end=end, interval=interval)
    data = data['Close']
    print(f"Fetched data length: {len(data)}")
    return data

def check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min):
    if effective_margin_max < effective_margin:
        effective_margin_max = effective_margin
    if effective_margin_min > effective_margin:
        effective_margin_min = effective_margin
    return effective_margin_max, effective_margin_min

def traripi_backtest(calculator, data, initial_funds, grid_start, grid_end, num_traps, profit_width, order_size, entry_interval=None, total_threshold=None,strategy='standard', density=1):
    """
    Perform Trailing Stop strategy backtest on given data.
    """
    margin_deposit = initial_funds
    effective_margin = margin_deposit
    effective_margin_max = -np.inf
    effective_margin_min = np.inf
    realized_profit = 0
    squared_realized_profit = 0
    realized_swap_point = 0
    squared_swap_point = 0
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
                                positions.append([order_size, i, 'Buy', grid, 0, add_required_margin,date])
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
                                positions.append([order_size, i, 'Buy', grid, 0, add_required_margin, date])
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
                        effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
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
                        effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                        margin_deposit += order_size * profit_width 
                        profit = order_size * profit_width
                        realized_profit += profit
                        squared_realized_profit += profit**2
                        required_margin -= pos[5]
                        pos[5] = 0
                        pos[2] = 'Buy-Closed'
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
            if margin_maintenance_flag:
                for pos in positions:
                    if pos[2] == 'Buy':
                        profit = (price - pos[3]) * order_size  # 現在の損失計算
                        effective_margin += profit - pos[4] # 損失分を証拠金に反映
                        effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                        margin_deposit += profit
                        realized_profit += profit
                        squared_realized_profit += profit**2
                        required_margin -= pos[5]
                        pos[5] = 0
                        pos[2] = "Buy-Closed"
                        trades.append((date, price, 'Forced Closed'))
                        print(f"Forced Closed at {price} with grid {pos[3]}, Effective Margin: {effective_margin}")
                        if abs(required_margin) > 0:
                            margin_maintenance_rate = effective_margin / required_margin * 100
                        else:
                            margin_maintenance_rate = float('inf')



                #""" 
                #check swap
            for pos in positions:
                if pos[2] == "Buy" or (pos[2] == "Buy-Closed" and add_business_days(pos[6],1) == date):
                    
                    realized_swap_point += calculator.get_total_swap_points(pair,pos[2],pos[6],date,order_size)
                    effective_margin += calculator.get_total_swap_points(pair,pos[2],pos[6],date,order_size)
                    squared_swap_point += calculator.get_total_swap_points(pair,pos[2],pos[6],date,order_size)**2
                    effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                    print(f'added swap to effective_margin: {effective_margin}')
                    if not "Closed" in pos[2]:
                        pos[6] = date

                        

                if abs(required_margin) > 0:
                    margin_maintenance_rate = effective_margin / required_margin * 100
                    if margin_maintenance_rate <= 100:
                        print("executed loss cut in check swap")
                        margin_maintenance_flag = True
                        continue
                else:
                    margin_maintenance_rate = float('inf')
                    #"""




    elif strategy == 'short_only':
        grids = np.linspace(grid_start, grid_end, num=num_traps)
        order_margin = sum(order_size * grid * required_margin_rate for grid in grids)
        order_capacity = effective_margin - (required_margin + order_margin)
        if order_capacity <= 0:
            print(f'cannot order because of lack of order_capacity')
            order_capacity_flag = True
            
        last_price = None  # Variable to store the last processed price

        for i in range(len(data)):
            if margin_maintenance_flag:
                break
            date = data.index[i]
            price = data.iloc[i]

            if last_price is not None:
                # Check if price has crossed any grid between last_price and price
                if price < last_price:
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
                                positions.append([order_size, i, 'Sell', grid, 0, add_required_margin,date])
                                trades.append((date, price, 'Sell'))
                                print(f"Opened Sell position at price {price}, last_price {last_price} with grid {grid}, Effective Margin: {effective_margin}, Required Margin:{required_margin}")
                                #break  # Exit loop once position is taken
                                if abs(required_margin) > 0:
                                    margin_maintenance_rate = effective_margin / required_margin * 100
                                    #print(f"Updated Margin Maintenance Rate in if last_price >= grid > price: {margin_maintenance_rate}")
                                    if margin_maintenance_rate <= 100:
                                        print("executed loss cut in if last_price >= grid > price")
                                        margin_maintenance_flag = True
                                        break
                                else:
                                    margin_maintenance_rate = float('inf')

                elif price > last_price:
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
                            if margin_maintenance_rate > 100 and order_capacity > 0:
                                #margin_deposit -= order_size * grid
                                #effective_margin -= order_size * grid
                                order_margin -= order_size * grid * required_margin_rate
                                add_required_margin = grid * order_size * required_margin_rate
                                required_margin += add_required_margin
                                positions.append([order_size, i, 'Sell', grid, 0, add_required_margin,date])
                                trades.append((date, price, 'Sell'))
                                print(f"Opened Sell position at price {price}, last_price {last_price} with grid {grid}, Effective Margin: {effective_margin}, Required Margin:{required_margin}")
                                #break  # Exit loop once position is taken

                                if abs(required_margin) > 0:
                                    margin_maintenance_rate = effective_margin / required_margin * 100
                                    #print(f"Updated Margin Maintenance Rate in if last_price <= grid < price: {margin_maintenance_rate}")
                                    if margin_maintenance_rate <= 100:
                                        print("executed loss cut in if last_price <= grid < price")
                                        margin_maintenance_flag = True
                                        break
                                else:
                                    margin_maintenance_rate = float('inf')


            # Position closure processing
            for pos in positions[:]:
                if pos[2] == 'Sell' and pos[1] < len(data) - 1:
                    if price - pos[3] > -profit_width:
                        # Update unrealized profit for open positions
                        unrealized_profit = order_size * (pos[3] - price)
                        effective_margin += unrealized_profit -  pos[4]  # Adjust for previous unrealized profit
                        effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                        add_required_margin = -pos[5] + price * order_size * required_margin_rate
                        required_margin += add_required_margin
                        pos[4] = unrealized_profit  # Store current unrealized profit in the position
                        pos[5] += add_required_margin
                        print(f"updated effective margin against price {price} , Effective Margin: {effective_margin}, Required Margin:{required_margin}, pos[5]:{pos[5]}")

                        if abs(required_margin) > 0:
                            margin_maintenance_rate = effective_margin / required_margin * 100
                            #print(f"Updated Margin Maintenance Rate in if price - pos[3] > -profit_width: {margin_maintenance_rate}")
                            if margin_maintenance_rate <= 100:
                                print(f"executed loss cut in 'if price - pos[3] > -profit_width' pos: {pos}, last postions: {positions[-1]}")
                                margin_maintenance_flag = True
                                continue
                        else:
                            margin_maintenance_rate = float('inf')

                    if price - pos[3] <=  - profit_width:
                        effective_margin += order_size * profit_width - pos[4]
                        effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                        margin_deposit += order_size * profit_width
                        profit = order_size * profit_width
                        realized_profit += profit
                        squared_realized_profit += profit**2
                        required_margin -= pos[5]
                        pos[5] = 0
                        pos[2] = 'Sell-Closed'
                        trades.append((date, price, 'Buy'))
                        print(f"Closed Buy position at {pos[3]+profit_width} with profit {profit} ,grid {pos[3]}, Effective Margin: {effective_margin}")
                        #break

                        if abs(required_margin) > 0:
                            margin_maintenance_rate = effective_margin / required_margin * 100
                            print(f"Updated Margin Maintenance Rate if price - pos[3] > -profit_width: {margin_maintenance_rate}")
                            if margin_maintenance_rate <= 100:
                                print("executed loss cut in price - pos[3] <= -profit_width")
                                margin_maintenance_flag = True
                                continue
                        else:
                            margin_maintenance_rate = float('inf')



            # Update last_price for the next iteration
            last_price = price


           # 強制ロスカットのチェック
            if margin_maintenance_flag:
                for pos in positions:
                    if pos[2] == 'Sell':
                        profit = - (price - pos[3]) * order_size  # 現在の損失計算
                        effective_margin += profit - pos[4] # 損失分を証拠金に反映
                        effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                        margin_deposit += profit
                        realized_profit += profit
                        squared_realized_profit += profit**2
                        required_margin -= pos[5]
                        #print(f'required_margin: {required_margin}')
                        pos[5] = 0
                        pos[2] = "Sell-Closed"
                        trades.append((date, price, 'Forced Closed'))
                        print(f"Forced Closed at {price} with grid {pos[3]}, Effective Margin: {effective_margin}, Required Margin:{required_margin}")
                        if abs(required_margin) > 0:
                            margin_maintenance_rate = effective_margin / required_margin * 100
                        else:
                            margin_maintenance_rate = float('inf')



                #""" 
                #check swap
            for pos in positions:
                if pos[2] == "Sell" or (pos[2] == "Sell-Closed" and add_business_days(pos[6],1) == date):

                    realized_swap_point += calculator.get_total_swap_points(pair,pos[2],pos[6],date,order_size)
                    effective_margin += calculator.get_total_swap_points(pair,pos[2],pos[6],date,order_size)
                    squared_swap_point += calculator.get_total_swap_points(pair,pos[2],pos[6],date,order_size)**2
                    effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                    print(f'added swap to effective_margin: {effective_margin}')
                    if not "Closed" in pos[2]:
                        pos[6] = date

                        

                if abs(required_margin) > 0:
                    margin_maintenance_rate = effective_margin / required_margin * 100
                    if margin_maintenance_rate <= 100:
                        print("executed loss cut in check swap")
                        margin_maintenance_flag = True
                        continue
                else:
                    margin_maintenance_rate = float('inf')
                    #"""




    elif strategy == 'half_and_half':
        half_point = (grid_start + grid_end) / 2
        grids_bottom = np.linspace(grid_start, half_point, num=int(num_traps / 2))
        grids_top = np.linspace(half_point, grid_end, num=int(num_traps / 2))
        
        order_margin = sum(order_size * grid * required_margin_rate for grid in np.concatenate((grids_bottom, grids_top)))
        order_capacity = effective_margin - (required_margin + order_margin)
        if order_capacity <= 0:
            print(f'cannot order because of lack of order_capacity')
            order_capacity_flag = True
        last_price = None  # Variable to store the last processed price

        for i in range(len(data)):
            if margin_maintenance_flag:
                break
            date = data.index[i]
            price = data.iloc[i]


            if last_price is not None:
                # Check bottom half area
                if price <= half_point:
                    for grid in grids_bottom:
                        if last_price > grid >= price:
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
                                positions.append([order_size, i, 'Buy', grid, 0, add_required_margin,date])
                                trades.append((date, price, 'Buy'))
                                print(f"Opened Buy position at {price} with grid {grid}, Effective Margin: {effective_margin}")
                                #break
                                if abs(required_margin) > 0:
                                    margin_maintenance_rate = effective_margin / required_margin * 100
                                    if margin_maintenance_rate <= 100:
                                        print("executed loss cut")
                                        margin_maintenance_flag = True
                                        break
                                else:
                                    margin_maintenance_rate = float('inf')


                        if last_price < grid <= price:
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
                                positions.append([order_size, i, 'Buy', grid, 0, add_required_margin,date])
                                trades.append((date, price, 'Buy'))
                                print(f"Opened Buy position at {price} with grid {grid}, Effective Margin: {effective_margin}")
                                #break
                                if abs(required_margin) > 0:
                                    margin_maintenance_rate = effective_margin / required_margin * 100
                                    if margin_maintenance_rate <= 100:
                                        print("executed loss cut")
                                        margin_maintenance_flag = True
                                        break
                                else:
                                    margin_maintenance_rate = float('inf')


                # Check top half area
                if price > half_point:
                    for grid in grids_top:
                        if last_price < grid <= price:
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
                                positions.append([order_size, i, 'Sell', grid, 0, add_required_margin,date])
                                trades.append((date, price, 'Sell'))
                                print(f"Opened Sell position at {price} with grid {grid}, Effective Margin: {effective_margin}")
                                #break
                                if abs(required_margin) > 0:
                                    margin_maintenance_rate = effective_margin / required_margin * 100
                                    if margin_maintenance_rate <= 100:
                                        print("executed loss cut")
                                        margin_maintenance_flag = True
                                        break
                                else:
                                    margin_maintenance_rate = float('inf')
                    
                        if last_price > grid >= price:
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
                                positions.append([order_size, i, 'Sell', grid, 0, add_required_margin,date])
                                trades.append((date, price, 'Sell'))
                                print(f"Opened Sell position at {price} with grid {grid}, Effective Margin: {effective_margin}")
                                #break

                                if abs(required_margin) > 0:
                                    margin_maintenance_rate = effective_margin / required_margin * 100
                                    if margin_maintenance_rate <= 100:
                                        print("executed loss cut")
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
                        effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                        add_required_margin = -pos[5] + price * order_size * required_margin_rate
                        required_margin += add_required_margin
                        pos[4] = unrealized_profit  # Store current unrealized profit in the position
                        pos[5] += add_required_margin
                        print(f"updated effective margin against price {price} , Effective Margin: {effective_margin}")
                        if abs(required_margin) > 0:
                            margin_maintenance_rate = effective_margin / required_margin * 100
                            if margin_maintenance_rate <= 100:
                                print("executed loss cut")
                                margin_maintenance_flag = True
                                continue
                        else:
                            margin_maintenance_rate = float('inf')

                    if price - pos[3] >=  profit_width:
                        effective_margin += order_size * profit_width -pos[4]
                        effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                        margin_deposit += order_size * profit_width
                        profit = order_size * profit_width
                        realized_profit += profit
                        squared_realized_profit += profit**2
                        required_margin -= pos[5]
                        pos[5] = 0
                        pos[2] = 'Buy-Closed'
                        trades.append((date, price, 'Sell'))
                        print(f"Closed Sell position at {pos[3]+profit_width} with profit {profit} ,grid {pos[3]}, Effective Margin: {effective_margin}")
                        #break
                        if abs(required_margin) > 0:
                            margin_maintenance_rate = effective_margin / required_margin * 100
                            if margin_maintenance_rate <= 100:
                                print("executed loss cut")
                                margin_maintenance_flag = True
                                continue
                        else:
                            margin_maintenance_rate = float('inf')

                elif pos[2] == 'Sell' and pos[1] < len(data) - 1:
                    if price - pos[3] > -profit_width:
                        # Update unrealized profit for open positions
                        unrealized_profit = order_size * (pos[3] - price)
                        effective_margin += unrealized_profit -  pos[4]  # Adjust for previous unrealized profit
                        effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                        add_required_margin = -pos[5] + price * order_size * required_margin_rate
                        required_margin += add_required_margin
                        pos[4] = unrealized_profit  # Store current unrealized profit in the position
                        pos[5] += add_required_margin
                        print(f"updated effective margin against price {price} , Effective Margin: {effective_margin}")

                        if abs(required_margin) > 0:
                            margin_maintenance_rate = effective_margin / required_margin * 100
                            if margin_maintenance_rate <= 100:
                                print("executed loss cut")
                                margin_maintenance_flag = True
                                continue
                        else:
                            margin_maintenance_rate = float('inf')

                    if price - pos[3] <=  - profit_width:
                        effective_margin += order_size * profit_width - pos[4]
                        effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                        margin_deposit += order_size * profit_width
                        profit = order_size * profit_width
                        realized_profit += profit
                        squared_realized_profit += profit**2
                        required_margin -= pos[5]
                        pos[5] = 0
                        pos[2] = 'Sell-Closed'
                        trades.append((date, price, 'Buy'))
                        print(f"Closed Buy position at {pos[3]+profit_width} with profit {profit} ,grid {pos[3]}, Effective Margin: {effective_margin}")
                        #break

                        if abs(required_margin) > 0:
                            margin_maintenance_rate = effective_margin / required_margin * 100
                            if margin_maintenance_rate <= 100:
                                print("executed loss cut")
                                margin_maintenance_flag = True
                                continue
                        else:
                            margin_maintenance_rate = float('inf')


            # Update last_price for the next iteration
            last_price = price



           # 強制ロスカットのチェック
            if margin_maintenance_flag:
                for pos in positions:
                    if pos[2] == 'Sell' or pos[2] == 'Buy':
                        if pos[2] == 'Sell':
                            profit = - (price - pos[3]) * order_size  # 現在の損失計算
                        if pos[2] == 'Buy':
                            profit = (price - pos[3]) * order_size
                        effective_margin += profit - pos[4] # 損失分を証拠金に反映
                        effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                        margin_deposit += profit
                        realized_profit += profit
                        squared_realized_profit += profit**2
                        required_margin -= pos[5]
                        pos[5] = 0
                        pos[2] += "-Closed"
                        trades.append((date, price, 'Forced Closed'))
                        print(f"Forced Closed at {price} with grid {pos[3]}, Effective Margin: {effective_margin}")
                        if abs(required_margin) > 0:
                            margin_maintenance_rate = effective_margin / required_margin * 100
                        else:
                            margin_maintenance_rate = float('inf')


                #""" 
                #check swap
            for pos in positions:
                if pos[2] == "Buy" or (pos[2] == "Buy-Closed" and add_business_days(pos[6],1) == date) or pos[2] == "Sell" or (pos[2] == "Sell-Closed" and add_business_days(pos[6],1) == date):

                    realized_swap_point += calculator.get_total_swap_points(pair,pos[2],pos[6],date,order_size)
                    effective_margin += calculator.get_total_swap_points(pair,pos[2],pos[6],date,order_size)
                    squared_swap_point += calculator.get_total_swap_points(pair,pos[2],pos[6],date,order_size)**2
                    effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                    print(f'added swap to effective_margin: {effective_margin}')
                    if not "Closed" in pos[2]:
                        pos[6] = date

                        

                if abs(required_margin) > 0:
                    margin_maintenance_rate = effective_margin / required_margin * 100
                    if margin_maintenance_rate <= 100:
                        print("executed loss cut in check swap")
                        margin_maintenance_flag = True
                        continue
                else:
                    margin_maintenance_rate = float('inf')
                    #"""







    elif strategy == 'diamond':
        quarter_point = (grid_start + grid_end) / 4
        half_point = (grid_start + grid_end) / 2
        three_quarter_point = 3 * (grid_start + grid_end) / 4

        # Set grid numbers for each area
        grids_bottom = np.linspace(grid_start, quarter_point, num=int(num_traps / 4))
        grids_lower_center = np.linspace(quarter_point, half_point, num=int(num_traps /4 * density))
        grids_upper_center = np.linspace(half_point, three_quarter_point, num=int(num_traps / 4 * density))
        grids_top = np.linspace(three_quarter_point, grid_end, num=int(num_traps / 4))
        
        order_margin = sum(order_size * grid * required_margin_rate for grid in np.concatenate((grids_bottom, grids_lower_center, grids_upper_center, grids_top)))
        order_capacity = effective_margin - (required_margin + order_margin)
        if order_capacity <= 0:
            print(f'cannot order because of lack of order_capacity')
            order_capacity_flag = True
            
        last_price = None  # Variable to store the last processed price


        for i in range(len(data)):
            if margin_maintenance_flag:
                break
            date = data.index[i]
            price = data.iloc[i]


            if last_price is not None:
                # Check bottom two areas
                if price <= half_point:
                    for grid in np.concatenate([grids_bottom, grids_lower_center]):
                        if last_price > grid >= price:
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
                                positions.append([order_size, i, 'Buy', grid, 0, add_required_margin,date])
                                trades.append((date, price, 'Buy'))
                                print(f"Opened Buy position at {price} with grid {grid}, Effective Margin: {effective_margin}")
                                #break
                                if abs(required_margin) > 0:
                                    margin_maintenance_rate = effective_margin / required_margin * 100
                                    if margin_maintenance_rate <= 100:
                                        print("executed loss cut")
                                        margin_maintenance_flag = True
                                        break
                                else:
                                    margin_maintenance_rate = float('Inf')


                        if last_price < grid <= price:
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
                                positions.append([order_size, i, 'Buy', grid, 0, add_required_margin,date])
                                trades.append((date, price, 'Buy'))
                                print(f"Opened Buy position at {price} with grid {grid}, Effective Margin: {effective_margin}")
                                #break
                                if abs(required_margin) > 0:
                                    margin_maintenance_rate = effective_margin / required_margin * 100
                                    if margin_maintenance_rate <= 100:
                                        print("executed loss cut")
                                        margin_maintenance_flag = True
                                        break
                                else:
                                    margin_maintenance_rate = float('inf')


                # Check top two areas
                if price >= half_point:
                    for grid in np.concatenate([grids_top, grids_upper_center]):
                        if last_price < grid <= price:
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
                                positions.append([order_size, i, 'Sell', grid, 0, add_required_margin,date])
                                trades.append((date, price, 'Sell'))
                                print(f"Opened Sell position at {price} with grid {grid}, Effective Margin: {effective_margin}, Required Margin: {required_margin}")
                                #break

                                if abs(required_margin) > 0:
                                    margin_maintenance_rate = effective_margin / required_margin * 100
                                    if margin_maintenance_rate <= 100:
                                        print("executed loss cut last_price < grid <= price")
                                        margin_maintenance_flag = True
                                        break
                                else:
                                    margin_maintenance_rate = float('inf')

                        if last_price > grid >= price:
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
                                positions.append([order_size, i, 'Sell', grid, 0, add_required_margin,date])
                                trades.append((date, price, 'Sell'))
                                print(f"Opened Sell position at {price} with grid {grid}, Effective Margin: {effective_margin}")
                                #break

                                if abs(required_margin) > 0:
                                    margin_maintenance_rate = effective_margin / required_margin * 100
                                    if margin_maintenance_rate <= 100:
                                        print("executed loss cut in last_price > grid >= price")
                                        margin_maintenance_flag = True
                                        break
                                else:
                                    float('inf')

            # Position closure processing
            for pos in positions[:]:
                if pos[2] == 'Buy' and pos[1] < len(data) - 1:
                    if price - pos[3] < profit_width:
                        # Update unrealized profit for open positions
                        unrealized_profit = order_size * (price - pos[3])
                        effective_margin += unrealized_profit -  pos[4]  # Adjust for previous unrealized profit
                        effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                        add_required_margin = -pos[5] + price * order_size * required_margin_rate
                        required_margin += add_required_margin
                        pos[4] = unrealized_profit  # Store current unrealized profit in the position
                        pos[5] += add_required_margin
                        print(f"updated effective margin against price {price} , Effective Margin: {effective_margin}")
                        if abs(required_margin) > 0:
                            margin_maintenance_rate = effective_margin / required_margin * 100
                            if margin_maintenance_rate <= 100:
                                print("executed loss cut price - pos[3] < profit_width")
                                margin_maintenance_flag = True
                                continue
                        else:
                            margin_maintenance_rate = float('inf')

                    if price - pos[3] >=  profit_width:
                        effective_margin += order_size * profit_width - pos[4]
                        effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                        margin_deposit += order_size * profit_width
                        profit = order_size * profit_width
                        realized_profit += profit
                        squared_realized_profit += profit**2
                        required_margin -= pos[5]
                        pos[5] = 0
                        pos[2] = 'Buy-Closed'
                        trades.append((date, price, 'Sell'))
                        print(f"Closed Sell position at {price} with profit {profit} ,grid {pos[3]}, Effective Margin: {effective_margin}")
                        #break
                        if abs(required_margin) > 0:
                            margin_maintenance_rate = effective_margin / required_margin * 100
                            if margin_maintenance_rate <= 100:
                                print("executed loss cut in price - pos[3] >= profit_width")
                                margin_maintenance_flag = True
                                continue
                        else:
                            margin_maintenance_rate = float('inf')

                elif pos[2] == 'Sell' and pos[1] < len(data) - 1:
                        if price - pos[3] > -profit_width:
                            # Update unrealized profit for open positions
                            unrealized_profit = order_size * (pos[3] - price)
                            effective_margin += unrealized_profit -  pos[4]  # Adjust for previous unrealized profit
                            effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                            add_required_margin = -pos[5] + price * order_size * required_margin_rate
                            required_margin += add_required_margin
                            pos[4] = unrealized_profit  # Store current unrealized profit in the                         position_value += 
                            pos[5] += add_required_margin
                            #print(f"updated effective margin against price {price} , Effective Margin: {effective_margin}")

                            if abs(required_margin) > 0:
                                margin_maintenance_rate = effective_margin / required_margin * 100
                                if margin_maintenance_rate <= 100:
                                    print("executed loss cut in price - pos[3] > -profit_width")
                                    margin_maintenance_flag = True
                                    continue
                            else:
                                margin_maintenance_rate = float('inf')

                        if price - pos[3] <=  - profit_width:
                            effective_margin += order_size * profit_width - pos[4]
                            effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                            margin_deposit += order_size * profit_width
                            profit = order_size * profit_width
                            realized_profit += profit
                            squared_realized_profit += profit**2
                            required_margin -= pos[5]
                            pos[5] = 0
                            pos[2] = 'Sell-Closed'
                            trades.append((date, price, 'Buy'))
                            print(f"Closed Buy position at {price} with profit {profit} ,grid {pos[3]}, Effective Margin: {effective_margin}")
                            #break

                            if abs(required_margin) > 0:
                                margin_maintenance_rate = effective_margin / required_margin * 100
                                if margin_maintenance_rate <= 100:
                                    print("executed loss cut in price - pos[3] <= -profit_width")
                                    margin_maintenance_flag = True
                                    continue
                            else:
                                margin_maintenance_rate = float('inf')


            # Update last_price for the next iteration
            last_price = price


            
           # 強制ロスカットのチェック
            if margin_maintenance_flag:
                for pos in positions:
                    if pos[2] == 'Sell' or pos[2] == 'Buy':
                        if pos[2] == 'Sell':
                            profit = - (price - pos[3]) * order_size  # 現在の損失計算
                        if pos[2] == 'Buy':
                            profit = (price - pos[3]) * order_size
                        effective_margin += profit - pos[4] # 損失分を証拠金に反映
                        effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                        margin_deposit += profit
                        realized_profit += profit
                        squared_realized_profit += profit**2
                        required_margin -= pos[5]
                        pos[5] = 0
                        pos[2] += "-Closed"
                        trades.append((date, price, 'Forced Closed'))
                        #print(f"Forced Closed at {price} with grid {pos[3]}, Effective Margin: {effective_margin}")
                        if abs(required_margin) > 0:
                            margin_maintenance_rate = effective_margin / required_margin * 100
                        else:
                            margin_maintenance_rate = float('inf')
              


                #""" 
                #check swap
            for pos in positions:
                if pos[2] == "Buy" or (pos[2] == "Buy-Closed" and add_business_days(pos[6],1) == date) or pos[2] == "Sell" or (pos[2] == "Sell-Closed" and add_business_days(pos[6],1) == date):

                    realized_swap_point += calculator.get_total_swap_points(pair,pos[2],pos[6],date,order_size)
                    effective_margin += calculator.get_total_swap_points(pair,pos[2],pos[6],date,order_size)
                    squared_swap_point += calculator.get_total_swap_points(pair,pos[2],pos[6],date,order_size)**2
                    effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                    print(f'added swap to effective_margin: {effective_margin}')
                    if not "Closed" in pos[2]:
                        pos[6] = date

                        

                if abs(required_margin) > 0:
                    margin_maintenance_rate = effective_margin / required_margin * 100
                    if margin_maintenance_rate <= 100:
                        print("executed loss cut in check swap")
                        margin_maintenance_flag = True
                        continue
                else:
                    margin_maintenance_rate = float('inf')
                    #"""





    elif strategy == 'milagroman' or strategy == 'milagroman2':
        last_price = None  # Variable to store the last processed price
        
        for i in range(len(data)):
            if margin_maintenance_flag:
                break
            date = data.index[i]
            price = data.iloc[i]

            if last_price is not None:
                # 買いと売りの同時エントリー
                add_required_margin = price * order_size * required_margin_rate
                required_margin += add_required_margin
                positions.append([order_size, i, 'Buy', price, 0, add_required_margin,date])
                add_required_margin = price * order_size * required_margin_rate
                required_margin += add_required_margin        
                positions.append([order_size, i, 'Sell', price, 0, add_required_margin,date])


                    # 売りポジションの決済
                for pos in positions:
                    if pos[2] == "Sell":
                        if (pos[3] - price >= profit_width and strategy == "milagroman") or (last_price - price >= profit_width and strategy == "milagroman2"):
                            effective_margin += order_size * (pos[3] - price) - pos[4]
                            effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                            margin_deposit += order_size * (pos[3] - price)
                            profit =  (pos[3] - price) * order_size
                            realized_profit += profit
                            squared_realized_profit += profit**2
                            required_margin -= pos[5]
                            pos[5] = 0
                            pos[2] = "Sell-Closed"
                            
                            if abs(required_margin) > 0:
                                margin_maintenance_rate = effective_margin / required_margin * 100
                                if margin_maintenance_rate <= 100:
                                    print("executed loss cut in last_price <= grid < price")
                                    margin_maintenance_flag = True
                                    break
                            else:
                                margin_maintenance_rate = float('inf')
                
                # 買いポジションのナンピン
                if price - last_price < - entry_interval:
                    add_required_margin = price * order_size * required_margin_rate
                    required_margin += add_required_margin
                    positions.append([order_size, i, 'Buy', price, 0, add_required_margin,date])

                    if abs(required_margin) > 0:
                        margin_maintenance_rate = effective_margin / required_margin * 100
                        if margin_maintenance_rate <= 100:
                            print("executed loss cut in last_price <= grid < price")
                            margin_maintenance_flag = True
                            break
                    else:
                        margin_maintenance_rate = float('inf')
                
                
                for pos in positions:
                    
                    if margin_deposit - effective_margin < total_threshold:	#margin_deposit - effective_margin = unrealized_profit
                        if pos[2] == 'Buy' and pos[1] < len(data) - 1:
                            # Update unrealized profit for open positions
                            unrealized_profit = order_size * (price - pos[3])
                            effective_margin += unrealized_profit -  pos[4]  # Adjust for previous unrealized profit
                            effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                            add_required_margin = -pos[5] + price * order_size * required_margin_rate
                            required_margin += add_required_margin
                            pos[4] = unrealized_profit  # Store current unrealized profit in the position
                            pos[5] += add_required_margin
                            print(f"updated effective margin against price {price} , Effective Margin: {effective_margin}")
                            if abs(required_margin) > 0:
                                margin_maintenance_rate = effective_margin / required_margin * 100
                                if margin_maintenance_rate <= 100:
                                    print("executed loss cut")
                                    margin_maintenance_flag = True
                                    continue
                            else:
                                margin_maintenance_rate = float('inf')
    
    
                        elif pos[2] == 'Sell' and pos[1] < len(data) - 1:
                            # Update unrealized profit for open positions
                            unrealized_profit = order_size * (pos[3] - price)
                            effective_margin += unrealized_profit -  pos[4]  # Adjust for previous unrealized profit
                            effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                            add_required_margin = -pos[5] + price * order_size * required_margin_rate
                            required_margin += add_required_margin
                            pos[4] = unrealized_profit  # Store current unrealized profit in the position
                            pos[5] += add_required_margin
                            print(f"updated effective margin against price {price} , Effective Margin: {effective_margin}")
        
                            if abs(required_margin) > 0:
                                margin_maintenance_rate = effective_margin / required_margin * 100
                                if margin_maintenance_rate <= 100:
                                    print("executed loss cut")
                                    margin_maintenance_flag = True
                                    continue
                            else:
                                margin_maintenance_rate = float('inf')

                    if margin_deposit - effective_margin >= total_threshold:# 全ポジションの決済条件
                        if pos[2] == 'Sell' or pos[2] == 'Buy':
    
                            if pos[2] == 'Sell':
                                profit = - (price - pos[3]) * order_size  # 現在の損失計算
                            if pos[2] == 'Buy':
                                profit = (price - pos[3]) * order_size
                            effective_margin += profit - pos[4] # 損失分を証拠金に反映
                            effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                            margin_deposit += profit
                            realized_profit += profit
                            squared_realized_profit += profit**2
                            required_margin -= pos[5]
                            pos[5] = 0
                            pos[2] += "-Closed"
                            trades.append((date, price, 'All Closed'))
                            print(f"All Closed at {price} with grid {pos[3]}, Effective Margin: {effective_margin}")
                            if abs(required_margin) > 0:
                                margin_maintenance_rate = effective_margin / required_margin * 100
                                if margin_maintenance_rate < 100:
                                    margin_maintenance_flag = True
                                    continue
                            else:
                                margin_maintenance_rate = float('inf')


                # Update last_price for the next iteration
            last_price = price
    
    
               # 強制ロスカットのチェック
            if margin_maintenance_flag:
                for pos in positions:
                    if pos[2] == 'Sell' or pos[2] == 'Buy':
                        if pos[2] == 'Sell':
                            profit = - (price - pos[3]) * order_size  # 現在の損失計算
                        if pos[2] == 'Buy':
                            profit = (price - pos[3]) * order_size
                        effective_margin += profit - pos[4] # 損失分を証拠金に反映
                        effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                        margin_deposit += profit
                        realized_profit += profit
                        squared_realized_profit += profit**2
                        required_margin -= pos[5]
                        pos[5] = 0
                        pos[2] += "-Closed"
                        trades.append((date, price, 'Forced Closed'))
                        print(f"Forced Closed at {price} with grid {pos[3]}, Effective Margin: {effective_margin}")
                        if abs(required_margin) > 0:
                            margin_maintenance_rate = effective_margin / required_margin * 100
                        else:
                            margin_maintenance_rate = float('inf')


            #""" 
            #check swap
            for pos in positions:
                if pos[2] == "Buy" or (pos[2] == "Buy-Closed" and add_business_days(pos[6],1) == date) or pos[2] == "Sell" or (pos[2] == "Sell-Closed" and add_business_days(pos[6],1) == date):

                    realized_swap_point += calculator.get_total_swap_points(pair,pos[2],pos[6],date,order_size)
                    effective_margin += realized_swap_point
                    squared_swap_point += realized_swap_point**2
                    effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                    print(f'added swap to effective_margin: {effective_margin}')
                    if not "Closed" in pos[2]:
                        pos[6] = date

                        

                if abs(required_margin) > 0:
                    margin_maintenance_rate = effective_margin / required_margin * 100
                    if margin_maintenance_rate <= 100:
                        print("executed loss cut in check swap")
                        margin_maintenance_flag = True
                        continue
                else:
                    margin_maintenance_rate = float('inf')
                    #"""


	
    elif strategy == 'milagroman3':
        last_price = None  # Variable to store the last processed price
        last_gradient = None
        SMA_200 = data.rolling(window=200).mean()
        Gradient = np.gradient(SMA_200)
        for i in range(len(data)):
            if margin_maintenance_flag:
                break
            date = data.index[i]
            price = data.iloc[i]
            gradient = Gradient[i]

            if last_price is not None and last_gradient is not None:
                # 買い子ポジションと売り子ポジションの同時エントリー
                add_required_margin = price * order_size * required_margin_rate
                required_margin += add_required_margin
                positions.append([order_size, i, 'Buy-child', price, 0, add_required_margin,date])
                add_required_margin = price * order_size * required_margin_rate
                required_margin += add_required_margin        
                positions.append([order_size, i, 'Sell-child', price, 0, add_required_margin,date])



                    # 売り子ポジション、買い子ポジションの決済
                for pos in positions:
                    if pos[2] == "Sell-child":
                        if last_price - price >= profit_width:
                            effective_margin += order_size * (pos[3] - price) - pos[4]
                            effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                            margin_deposit += order_size * (pos[3] - price)
                            profit =  (pos[3] - price) * order_size
                            realized_profit += profit
                            squared_realized_profit += profit**2
                            required_margin -= pos[5]
                            pos[2] = "Sell-child-Closed"
                            
                            if abs(required_margin) > 0:
                                margin_maintenance_rate = effective_margin / required_margin * 100
                                if margin_maintenance_rate <= 100:
                                    print("executed loss cut in last_price <= grid < price")
                                    margin_maintenance_flag = True
                                    break
                            else:
                                margin_maintenance_rate = float('inf')
                        
                    if pos[2] == "Buy-child":
                        if price - last_price >= profit_width:
                            effective_margin += order_size * (price - pos[3] ) - pos[4]
                            effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                            margin_deposit += order_size * (price - pos[3])
                            profit =  (price - pos[3]) * order_size
                            realized_profit += profit
                            squared_realized_profit += profit**2
                            required_margin -= pos[5]
                            pos[2] = "Buy-child-Closed"
                            
                            if abs(required_margin) > 0:
                                margin_maintenance_rate = effective_margin / required_margin * 100
                                if margin_maintenance_rate <= 100:
                                    print("executed loss cut in last_price <= grid < price")
                                    margin_maintenance_flag = True
                                    break
                            else:
                                margin_maintenance_rate = float('inf')

                # 買いメインポジション、売りヘッジポジションのナンピン
                if price - last_price < - entry_interval and gradient > 0:
                    add_required_margin = price * order_size * required_margin_rate
                    required_margin += add_required_margin
                    positions.append([order_size, i, 'Buy-main', price, 0, add_required_margin,date])

                    if abs(required_margin) > 0:
                        margin_maintenance_rate = effective_margin / required_margin * 100
                        if margin_maintenance_rate <= 100:
                            print("executed loss cut in last_price <= grid < price")
                            margin_maintenance_flag = True
                            break
                    else:
                        margin_maintenance_rate = float('inf')
                        
                if price - last_price > entry_interval and gradient < 0:
                    add_required_margin = price * order_size * required_margin_rate
                    required_margin += add_required_margin
                    positions.append([order_size, i, 'Sell-hedge', price, 0, add_required_margin,date])

                    if abs(required_margin) > 0:
                        margin_maintenance_rate = effective_margin / required_margin * 100
                        if margin_maintenance_rate <= 100:
                            print("executed loss cut in last_price <= grid < price")
                            margin_maintenance_flag = True
                            break
                    else:
                        margin_maintenance_rate = float('inf')



                for pos in positions:
                    
                    if margin_deposit - effective_margin < total_threshold:	#margin_deposit - effective_margin = unrealized_profit
                        if (pos[2] == 'Buy-child' or pos[2] == 'Buy-main') and pos[1] < len(data) - 1:
                            # Update unrealized profit for open positions
                            unrealized_profit = order_size * (price - pos[3])
                            effective_margin += unrealized_profit -  pos[4]  # Adjust for previous unrealized profit
                            effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                            add_required_margin = -pos[5] + price * order_size * required_margin_rate
                            required_margin += add_required_margin
                            pos[4] = unrealized_profit  # Store current unrealized profit in the position
                            pos[5] += add_required_margin
                            print(f"updated effective margin against price {price} , Effective Margin: {effective_margin}")
                            if abs(required_margin) > 0:
                                margin_maintenance_rate = effective_margin / required_margin * 100
                                if margin_maintenance_rate <= 100:
                                    print("executed loss cut")
                                    margin_maintenance_flag = True
                                    continue
                            else:
                                margin_maintenance_rate = float('inf')
    
    
                        elif (pos[2] == 'Sell-child' or pos[2] == 'Sell-hedge') and pos[1] < len(data) - 1:
                            # Update unrealized profit for open positions
                            unrealized_profit = order_size * (pos[3] - price)
                            effective_margin += unrealized_profit -  pos[4]  # Adjust for previous unrealized profit
                            effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                            add_required_margin = -pos[5] + price * order_size * required_margin_rate
                            required_margin += add_required_margin
                            pos[4] = unrealized_profit  # Store current unrealized profit in the position
                            pos[5] += add_required_margin
                            print(f"updated effective margin against price {price} , Effective Margin: {effective_margin}")
        
                            if abs(required_margin) > 0:
                                margin_maintenance_rate = effective_margin / required_margin * 100
                                if margin_maintenance_rate <= 100:
                                    print("executed loss cut")
                                    margin_maintenance_flag = True
                                    continue
                            else:
                                margin_maintenance_rate = float('inf')

                    if margin_deposit - effective_margin >= total_threshold:# 全ポジションの決済条件
                        if pos[2] == 'Sell-child' or pos[2] == 'Sell-hedge' or pos[2] == 'Buy-child' or pos[2] == 'Buy-main':
    
                            if pos[2] == 'Sell-child' or pos[2] == 'Sell-hedge':
                                profit = - (price - pos[3]) * order_size  # 現在の損失計算
                            if pos[2] == 'Buy-child' or pos[2] == 'Buy-main':
                                profit = (price - pos[3]) * order_size
                            effective_margin += profit - pos[4] # 損失分を証拠金に反映
                            effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                            margin_deposit += profit
                            realized_profit += profit
                            squared_realized_profit += profit**2
                            required_margin -= pos[5]
                            pos[5] = 0
                            pos[2] += pos[2] + "-Closed"
                            trades.append((date, price, 'Forced Closed'))
                            print(f"All Closed at {price} with grid {pos[3]}, Effective Margin: {effective_margin}")
                            if abs(required_margin) > 0:
                                margin_maintenance_rate = effective_margin / required_margin * 100
                                if margin_maintenance_rate < 100:
                                    margin_maintenance_flag = True
                                    continue
                            else:
                                margin_maintenance_rate = float('inf')

                #売りヘッジポジションのトレンド判定による決済
                if gradient >0 and last_gradient < 0:
                    for pos in positions:
                        if pos[2] == 'Sell-hedge':
                            profit = - (price - pos[3]) * order_size  # 現在の損失計算
                            effective_margin += profit - pos[4] # 損失分を証拠金に反映
                            effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                            margin_deposit += profit
                            realized_profit += profit
                            squared_realized_profit += profit**2
                            required_margin -= pos[5]
                            pos[5] = 0
                            pos[2] += "-Closed"
                            print(f"Closed sell positions at price: {price}, gradient: {gradient} last_gradient: {last_gradient}, Effective Margin: {effective_margin}")
                            if abs(required_margin) > 0:
                                margin_maintenance_rate = effective_margin / required_margin * 100
                            else:
                                margin_maintenance_rate = float('inf')


                # Update last_price for the next iteration
            last_price = price
    

                # Update last_gradient for the next iteration
            last_gradient = gradient
    
               # 強制ロスカットのチェック
            if margin_maintenance_flag:
                for pos in positions:
                    if pos[2] == 'Sell-child' or pos[2] == 'Sell-hedge' or pos[2] == 'Buy-child' or pos[2] == 'Buy-main':
                        if pos[2] == 'Sell-child' or pos[2] == 'Sell-hedge':
                            profit = - (price - pos[3]) * order_size  # 現在の損失計算
                        if pos[2] == 'Buy-child' or 'Buy-main':
                            profit = (price - pos[3]) * order_size
                        effective_margin += profit - pos[4] # 損失分を証拠金に反映
                        effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                        margin_deposit += profit
                        realized_profit += profit
                        squared_realized_profit += profit**2
                        required_margin -= pos[5]
                        pos[5] = 0
                        pos[2] += "-Closed"
                        trades.append((date, price, 'Forced Closed'))
                        print(f"Forced Closed at {price} with grid {pos[3]}, Effective Margin: {effective_margin}")
                        if abs(required_margin) > 0:
                            margin_maintenance_rate = effective_margin / required_margin * 100
                        else:
                            margin_maintenance_rate = float('inf')


                #"""
            for pos in positions:
                #check swap
                if ("Buy" in pos[2] and not pos[2].endswith('Closed')) or ("Sell" in pos[2] and not pos[2].endswith('Closed')) or ("Closed" in pos[2] and add_business_days(pos[6],1) == date):

                    effective_margin += calculator.get_total_swap_points(pair,pos[2],pos[6],date,order_size)
                    squared_swap_point += calculator.get_total_swap_points(pair,pos[2],pos[6],date,order_size)**2
                    effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)
                    print(f'added swap to effective_margin: {effective_margin}')

                    if not "Closed" in pos[2]:
                        pos[6] = date
                    

                if abs(required_margin) > 0:
                    margin_maintenance_rate = effective_margin / required_margin * 100
                    if margin_maintenance_rate <= 100:
                        print("executed loss cut in check swap")
                        margin_maintenance_flag = True
                        continue
                else:
                    margin_maintenance_rate = float('inf')
                #"""



    if positions:
      #print(positions)
      buy_count = sum(1 if 'Buy' in status and not status.endswith('Closed') else 0 for size, index, status, _, _, _, _ in positions)
      sell_count = sum(1 if 'Sell' in status and not status.endswith('Closed') else 0 for size, index, status, _, _, _, _ in positions)
      buy_closed_count = sum(1 if "Buy" in status and 'Closed' in status  else 0 for size, index, status, _, _, _, _ in positions)
      sell_closed_count = sum(1 if "Sell" in status and 'Closed' in status else 0 for size, index, status, _, _, _, _ in positions)
      
      print(f'buy_count{buy_count}')
      print(f'sell_count{sell_count}')
      print(f'buy_closed_count{buy_closed_count}')
      print(f'sell_closed_count{sell_closed_count}')


    # Calculate position value
    if positions:
        position_value = sum(size * (data.iloc[-1] - grid) if 'Buy' in status and not status.endswith('Closed') else
                     -size * (data.iloc[-1] - grid) if 'Sell'  in status and not status.endswith('Closed') else
                     0 for size, _, status, grid, _, _, _ in positions)
    else:
        position_value = 0

    # Calculate swap values  
    #"""
    if positions:
        swap_value = sum(calculator.get_total_swap_points(pair,status,data.index[index],date,size) if ('Buy' in status or 'Sell' in status) and not status.endswith('Closed') else
                     0 for size, index, status, _, _, _, _ in positions) + sum(calculator.get_total_swap_points(pair,status,data.index[index],add_business_days(swap_day,1),size) if 'Closed' in status else 0 for size, index, status, _, _, _, swap_day in positions)
    else:
        swap_value = 0

    # Calculate sharp ratio
    if np.sqrt((squared_realized_profit+squared_swap_point)/len(data) - ((realized_profit+realized_swap_point)/len(data))**2) != 0:
        sharp_ratio = ((realized_profit+realized_swap_point)/len(data)) / np.sqrt((squared_realized_profit+squared_swap_point)/len(data) - ((realized_profit+realized_swap_point)/len(data))**2)
    else:
        sharp_ratio = None

    #Calculate max draw down
    max_draw_down = (effective_margin_max - effective_margin_min) / effective_margin_max * 100

    # Calculate margin deposit
    #margin_deposit = initial_funds + realized_profit

    return effective_margin, margin_deposit, realized_profit, position_value, swap_value, required_margin, margin_maintenance_rate, entry_interval, total_threshold, trades, sharp_ratio, max_draw_down



pair = 'USDJPY=X'
interval="1d"
end_date = datetime.strptime("2022-01-01","%Y-%m-%d")#datetime.now() - timedelta(days=7)
start_date = datetime.strptime("2010-01-01","%Y-%m-%d")#datetime.now() - timedelta(days=14)
initial_funds = 2000000
grid_start = 100
grid_end = 150
strategies = ['long_only','short_only', 'half_and_half', 'diamond']
entry_intervals = [-15]  # エントリー間隔
total_thresholds = [100]  # 全ポジション決済の閾値
# データの取得
data = fetch_currency_data(pair, start_date, end_date,interval)

if __name__ == "__main__":
    # パラメータ設定
    order_sizes = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
    num_traps_options = [20, 40, 60, 67, 80, 100]
    profit_widths = [1,2,3,4,5,6,7,8,9,9.52, 10]
    densities = [2,4,6,8,8.37,10]
    url = 'https://fx.minkabu.jp/hikaku/moneysquare/spreadswap.html'
    html = get_html(url)
    #print(html[:1000])  # デバッグ出力：取得したHTMLの先頭部分を表示
    swap_points = parse_swap_points(html)
    swap_points = rename_swap_points(swap_points)
    calculator = SwapCalculator(swap_points)
    
    results = []
    
    
    # バックテストの実行
    milagroman_list = [m for m in strategies if "milagroman" in m]
    if "diamond" in strategies and milagroman_list:
        print("hello diamond and {} both".format(milagroman_list))
        for order_size, num_traps, profit_width, strategy, density, entry_interval, total_threshold in product(order_sizes, num_traps_options, profit_widths, strategies, densities, entry_intervals, total_thresholds):
            effective_margin, margin_deposit, realized_profit, position_value, swap_value, required_margin, margin_maintenance_rate, entry_interval, total_threshold, trades, sharp_ratio, max_draw_down = traripi_backtest(
               calculator ,data, initial_funds, grid_start, grid_end, num_traps, profit_width, order_size, entry_interval, total_threshold, strategy=strategy, density=density
            )
      
            results.append((effective_margin, margin_deposit, realized_profit, position_value, swap_value, order_size, num_traps, profit_width, strategy, density, required_margin, margin_maintenance_rate, entry_interval, total_threshold, sharp_ratio, max_draw_down))
            
    elif "diamond" in strategies and not milagroman_list:
        print("hello diamond only")
        for order_size, num_traps, profit_width, strategy, density in product(order_sizes, num_traps_options, profit_widths, strategies, densities):
            effective_margin, margin_deposit, realized_profit, position_value, swap_value, required_margin, margin_maintenance_rate, _, _, trades, sharp_ratio, max_draw_down = traripi_backtest(
                calculator, data, initial_funds, grid_start, grid_end, num_traps, profit_width, order_size, entry_interval=None, total_threshold=None, strategy=strategy, density=density
            )
      
            results.append((effective_margin, margin_deposit, realized_profit, position_value, swap_value, order_size, num_traps, profit_width, strategy, density, required_margin, margin_maintenance_rate, None, None, sharp_ratio, max_draw_down
                            ))
    
    
    elif not "diamond" in strategies and milagroman_list:
        print("{} only".format(milagroman_list))
        for order_size, num_traps, profit_width, strategy, entry_interval, total_threshold in product(order_sizes, num_traps_options, profit_widths, strategies, entry_intervals, total_thresholds):
            effective_margin, margin_deposit, realized_profit, position_value, swap_value, required_margin, margin_maintenance_rate, entry_interval, total_threshold, trades, sharp_ratio, max_draw_down = traripi_backtest(
                calculator, data, initial_funds, grid_start, grid_end, num_traps, profit_width, order_size, entry_interval, total_threshold, strategy=strategy, density=None
            )
      
            results.append((effective_margin, margin_deposit, realized_profit, position_value, swap_value, order_size, num_traps, profit_width, strategy, None, required_margin, margin_maintenance_rate, entry_interval, total_threshold, sharp_ratio, max_draw_down))
    
    
    elif not "diamond" in strategies and not milagroman_list:
        print("nothing")
        for order_size, num_traps, profit_width, strategy in product(order_sizes, num_traps_options, profit_widths, strategies):
            effective_margin, margin_deposit, realized_profit, position_value, swap_value, required_margin, margin_maintenance_rate, entry_interval, total_threshold, trades, sharp_ratio, max_draw_down = traripi_backtest(
                calculator, data, initial_funds, grid_start, grid_end, num_traps, profit_width, order_size, None, None, strategy=strategy, density=None
            )
      
            results.append((effective_margin, margin_deposit, realized_profit, position_value, swap_value, order_size, num_traps, profit_width, strategy, None, required_margin, margin_maintenance_rate, None, None, sharp_ratio, max_draw_down))
    
        
    
    # 結果の表示
    results_df = pd.DataFrame(results, columns=[
        'Effective Margin', 'Margin Deposit', 'Realized Profit', 'Position Value', 'Swap Value' ,'Order Size', 'Num Traps', 'Profit Width', 'Strategy', 'Density','Required Margin', 'Margin Maintenance Rate', 'Entry Interval', 'Total Threshold', 'Sharp Ratio', 'Max Draw Down'
    ])
    
    # 結果の表示
    # ユニークな組み合わせを取得
    unique_results = results_df.drop_duplicates(subset=['Order Size', 'Num Traps', 'Profit Width', 'Strategy', 'Density', 'Entry Interval', 'Total Threshold', 'Sharp Ratio', 'Max Draw Down'])
    
    # Top 5 Results Based on Effective Margin
    print("上位5件の有効証拠金に基づく結果:")
    rank = 1
    seen_results = set()  # 重複を管理するためのセット
    for i, row in results_df.sort_values(by='Effective Margin', ascending=False).iterrows():
        key = (row['Margin Deposit'], row['Effective Margin'], row['Position Value'], row['Swap Value'], row['Realized Profit'], row['Sharp Ratio'], row['Max Draw Down'])
        if key in seen_results:
            continue
        seen_results.add(key)
        print(f"Rank {rank}:")
        print(f"  預託証拠金: {row['Margin Deposit']}")
        print(f"  有効証拠金: {row['Effective Margin']}")
        print(f"  ポジション損益: {row['Position Value']}")
        print(f"  スワップ損益: {row['Swap Value']}")
        print(f"  確定利益: {row['Realized Profit']}")
        print(f" 必要証拠金: {row['Required Margin']}")
        print(f"証拠金維持率: {row['Margin Maintenance Rate']}")
        print(f"シャープレシオ: {row['Sharp Ratio']}")
        print(f"最大ドローダウン: {row['Max Draw Down']}%")
        print(f"  取引通貨量: {row['Order Size']}, トラップ本数: {row['Num Traps']}, 利益値幅: {row['Profit Width']}, 戦略: {row['Strategy']}, 密度: {row['Density']}, エントリー間隔: {row['Entry Interval']}, 全ポジション決済の閾値: {row['Total Threshold']}")
        rank += 1
        if rank > 5:
            break
    
    print("\n最悪の3件の有効証拠金に基づく結果:")
    rank = 1
    seen_results = set()  # 重複を管理するためのセット
    for i, row in results_df.sort_values(by='Effective Margin').head(3).iterrows():
        key = (row['Margin Deposit'], row['Effective Margin'], row['Position Value'], row['Swap Value'], row['Realized Profit'])
        if key in seen_results:
            continue
        seen_results.add(key)
        print(f"Rank {rank}:")
        print(f"  預託証拠金: {row['Margin Deposit']}")
        print(f"  有効証拠金: {row['Effective Margin']}")
        print(f"  ポジション損益: {row['Position Value']}")
        print(f"  スワップ損益: {row['Swap Value']}")
        print(f"  確定利益: {row['Realized Profit']}")
        print(f" 必要証拠金: {row['Required Margin']}")
        print(f"証拠金維持率: {row['Margin Maintenance Rate']}")
        print(f"シャープレシオ: {row['Sharp Ratio']}")
        print(f"最大ドローダウン: {row['Max Draw Down']}%")
        print(f"  取引通貨量: {row['Order Size']}, トラップ本数: {row['Num Traps']}, 利益値幅: {row['Profit Width']}, 戦略: {row['Strategy']}, 密度: {row['Density']}, エントリー間隔: {row['Entry Interval']}, 全ポジション決済の閾値: {row['Total Threshold']}")
        rank += 1
    
    print(f"check total:{row['Margin Deposit']+row['Position Value']+row['Swap Value']}")
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