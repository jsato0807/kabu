import numpy as np
from multiprocessing import Pool, Manager, Lock
import multiprocessing
from itertools import product
from kabu_swap import SwapCalculator
from kabu_library import fetch_currency_data
from datetime import datetime
import pandas as pd
import holidays



def process_grid(date, grid, last_price, price, effective_margin, positions, order_size, i, required_margin_rate, required_margin, margin_maintenance_rate, order_margin, order_capacity_flag, margin_maintenance_flag, lock):
    if min(last_price, price) <= grid <= max(last_price, price):
        with lock:
            # 証拠金維持率を確認
            if margin_maintenance_rate <= 100:
                margin_maintenance_flag.value = True
                return

            # 発注容量を確認
            order_capacity = effective_margin.value - (required_margin.value + order_margin.value)
            if order_capacity < 0:
                order_capacity_flag.value = True
                print('発注容量が不足しているため、発注できません。')
                return

            # ポジションをオープンする
            if margin_maintenance_rate > 100 and order_capacity > 0:
                subtract_order_margin = order_size * grid * required_margin_rate
                add_required_margin = grid * order_size * required_margin_rate
                order_margin.value -= subtract_order_margin
                required_margin.value += add_required_margin
                positions.append([order_size, i, 'Buy', grid, 0, add_required_margin, date, 0, 0])
                print(f"グリッド {grid} で Buy ポジションをオープンしました。{required_margin.value}")

def process_grids_parallel(manager, date, grids, last_price, price, effective_margin, positions, order_size, i, required_margin_rate, margin_maintenance_rate,margin_maintenance_flag):
    # ロックオブジェクトを作成
    lock = manager.Lock()
    
    # 共有リソースを Manager で作成
    effective_margin = manager.Value('d', effective_margin)
    order_margin = manager.Value('d', 0.0)
    required_margin = manager.Value('d', 0.0)
    positions = manager.list(positions)  # ポジションリスト
    order_capacity_flag = manager.Value('b', False)
    margin_maintenance_flag = manager.Value('b', margin_maintenance_flag)
    
    # 並列処理用のプールを作成
    with Pool() as pool:
        pool.starmap(process_grid, [(date, grid, last_price, price, effective_margin, positions, order_size, i, required_margin_rate, required_margin, margin_maintenance_rate, order_margin, order_capacity_flag, margin_maintenance_flag, lock) for grid in grids])
    print(f"check positions in def process_grids_parallel: {list(positions)}")
    return order_margin.value, required_margin.value, list(positions), order_capacity_flag.value, margin_maintenance_flag.value



def process_position(data, margin_maintenance_flag, effective_margin, margin_deposit, realized_profit, required_margin, positions, index, price, profit_width, order_size, required_margin_rate, lock):
    """各ポジションの状態を処理する関数"""
    pos = positions[index]  # indexでアクセス
    if pos[2] == 'Buy' and pos[1] <= len(data) - 1:
        if price - pos[3] < profit_width:
            unrealized_profit = order_size * (price - pos[3])
            add_required_margin = -pos[5] + price * order_size * required_margin_rate
            with lock:
                effective_margin.value += unrealized_profit - pos[4]
                required_margin.value += add_required_margin
            pos[5] += add_required_margin
            pos[4] = unrealized_profit

        if price - pos[3] >= profit_width:
            profit = order_size * profit_width
            add_effective_margin = profit - pos[4]
            with lock:
                effective_margin.value += add_effective_margin
                margin_deposit.value += profit
                realized_profit.value += profit
                required_margin.value -= pos[5]
            pos[7] += profit
            pos[5] = 0
            pos[2] = 'Buy-Closed'

    # 更新した pos を positions に反映
    positions[index] = pos


def process_positions_parallel(manager, data, positions, effective_margin, margin_deposit, realized_profit, required_margin, margin_maintenance_flag, price, profit_width, order_size, required_margin_rate):
    # ロックオブジェクトを作成
    lock = manager.Lock()

    # 共有リソースを Manager で作成
    effective_margin = manager.Value('d', effective_margin)
    required_margin = manager.Value('d', required_margin)
    positions = manager.list(positions)  # ポジションリスト
    margin_maintenance_flag = manager.Value('b', margin_maintenance_flag)
    margin_deposit = manager.Value('d', margin_deposit)
    realized_profit = manager.Value('d', realized_profit)

    # 並列処理を行うために、pool.starmapを使用
    with Pool() as pool:
        pool.starmap(process_position, [
            (data, margin_maintenance_flag, effective_margin, margin_deposit, realized_profit, required_margin, positions, index, price, profit_width, order_size, required_margin_rate, lock)
            for index in range(len(positions))
        ])

    print(f"check effective_margin in def process_positions_parallel: {effective_margin.value}")
    print(f"check positions in def process_positions_parallel: {list(positions)}")

    return list(positions), effective_margin.value, margin_deposit.value, realized_profit.value, required_margin.value, margin_maintenance_flag.value


def process_swap(calculator, pair, order_size, data, pos, date, effective_margin, num_positions, interval, lock):
    """スワップ処理を行う関数"""
    if pos[2] == "Buy" or (pos[2] == "Buy-Closed" and calculator.add_business_days(pos[6],1,data.index,interval,pair) == date and data.index[pos[1]] != pos[6]): #last condition acts when a position is opend and closed in intraday
        with lock:
            add_effective_margin = calculator.get_total_swap_points(pair, pos[2], pos[6], date, order_size, data.index)
            #print(add_effective_margin)
            effective_margin.value += add_effective_margin
            num_positions.value += 1
            print(f'スワップが有効証拠金に追加されました: effective_margin: {effective_margin}, add_effective_margin: {add_effective_margin}, date: {date}, pos[6]: {pos[6]}')
        pos[8] += calculator.get_total_swap_points(pair,pos[2],pos[6],date,order_size,data.index)
        if "Closed" not in pos[2]:
            pos[6] = date

def process_swap_parallel(manager,calculator, pair, order_size, data, positions, date, num_positions, effective_margin, margin_maintenance_flag, interval):
    """並列でスワップ処理を行う関数"""
    # ロックオブジェクトを作成
    lock = manager.Lock()
    
    positions = manager.list(positions)
    effective_margin = manager.Value('d', effective_margin)
    margin_maintenance_flag = manager.Value('b', margin_maintenance_flag)
    num_positions = manager.Value('d', num_positions)

    with Pool() as pool:
        pool.starmap(process_swap, [(calculator, pair, order_size, data, pos, date, effective_margin, num_positions, interval, lock) for pos in positions])


    print(f"check positions in def process_swap_parallel: {list(positions)}")
    return effective_margin.value, num_positions.value, margin_maintenance_flag.value, list(positions)


def process_losscut(margin_maintenance_flag, index, positions, price, effective_margin, margin_deposit, realized_profit, required_margin, order_size, lock):
    pos = positions[index]
    if margin_maintenance_flag.value:
        if pos[2] == 'Buy':
            profit = (price - pos[3]) * order_size  # 現在の損失計算
            with lock:
                effective_margin.value += profit - pos[4] # 損失分を証拠金に反映
                margin_deposit.value += profit
                realized_profit.value += profit
                required_margin.value -= pos[5]
            pos[7] += profit
            pos[5] = 0
            pos[2] = "Buy-Closed"
            #trades.append((date, price, 'Forced Closed'))
            print(f"Forced Closed at {price} with grid {pos[3]}, Effective Margin: {effective_margin}")

    # 更新した pos を positions に反映
    positions[index] = pos

def process_losscut_parallel(manager, margin_maintenance_flag, positions, price, effective_margin, margin_deposit, realized_profit, required_margin, order_size):
    # ロックオブジェクトを作成
    lock = manager.Lock()
    
    effective_margin = manager.Value('d', effective_margin)
    margin_deposit = manager.Value('d', margin_deposit)
    realized_profit = manager.Value('d' , realized_profit)
    required_margin = manager.Value('d', required_margin)
    margin_maintenance_flag = manager.Value('b', margin_maintenance_flag) 
    positions = manager.list(positions)  

    with Pool() as pool:
        pool.starmap(process_losscut, [
            (margin_maintenance_flag,index, positions, price, effective_margin, margin_deposit, realized_profit, required_margin, order_size, lock) 
            for index in range(len(positions))
        ])

    print(f"check positions in def process_losscut_parallel: {list(positions)}")
    return effective_margin.value, margin_deposit.value, realized_profit.value, required_margin.value , margin_maintenance_flag.value, list(positions)


def update_margin_maintenance_rate(effective_margin, required_margin):
    if required_margin != 0:  
        margin_maintenance_rate = (effective_margin / required_margin) * 100
    else:
        margin_maintenance_rate = np.inf

    if margin_maintenance_rate <= 100:
        print(f"Margin maintenance rate is {margin_maintenance_rate}%, below threshold. Forced liquidation triggered.")
        return True, margin_maintenance_rate  # フラグと値を返す
    return False, margin_maintenance_rate  # フラグと値を返す


def traripi_backtest(calculator, data, initial_funds, grid_start, grid_end, num_traps, profit_width, order_size, entry_interval=None, total_threshold=None, strategy='standard', density=1,realized_profit=0, required_margin=0, position_value=0, swap_value=0, effective_margin_max = -np.inf, effective_margin_min = np.inf):  
    """トラリピバックテストを実行する関数"""
    RETURN = []
    margin_deposit = initial_funds + realized_profit
    margin_maintenance_rate = float('inf')
    required_margin_rate = 0.04
    order_margin = sum(order_size * grid * required_margin_rate for grid in np.linspace(grid_start, grid_end, num=num_traps))
    effective_margin = margin_deposit + position_value + swap_value
    margin_maintenance_flag = False
    order_capacity_flag = False
    positions = []
    num_positions = 0


    grids = np.linspace(grid_start, grid_end, num=num_traps)

    last_price = None
    with Manager() as manager:
        for i in range(len(data)):
            if margin_maintenance_flag or order_capacity_flag:
                break
            date = data.index[i]
            price = data.iloc[i]

            if last_price is not None:
                if price != last_price:
                    margin_maintenance_flag, margin_maintenance_rate = update_margin_maintenance_rate(effective_margin,required_margin)
                    if margin_maintenance_flag:
                        break
                    order_capacity = effective_margin - (required_margin + order_margin)
                    if order_capacity < 0:
                        order_capacity_flag = True
                        print(f'cannot order because of lack of order capacity')
                        break
                    if margin_maintenance_rate > 100 and order_capacity > 0:
                        order_margin, required_margin, positions, order_capacity_flag, margin_maintenance_flag = process_grids_parallel(
                            manager, date, grids, last_price, price, effective_margin, positions, order_size, i, required_margin_rate, margin_maintenance_rate, margin_maintenance_flag
                        )
                        margin_maintenance_flag, margin_maintenance_rate = update_margin_maintenance_rate(effective_margin, required_margin)
                        if margin_maintenance_flag:
                            print("executed loss cut in last_price <= grid < price")
                            #break                            
                    # ポジション処理
                    #print(positions)
                    positions, effective_margin, margin_deposit, realized_profit, required_margin, margin_maintenance_flag = process_positions_parallel(
                        manager, data, positions, effective_margin, margin_deposit, realized_profit, required_margin, margin_maintenance_flag, price, profit_width, order_size, required_margin_rate
                    )
                    print(f"check effective_margin after position process: {effective_margin}")
                    effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)

                    margin_maintenance_flag, margin_maintenance_rate = update_margin_maintenance_rate(effective_margin, required_margin)
                    if margin_maintenance_flag:
                            print("executed loss cut in position closure")
                            #break

                    # スワップ処理
                    effective_margin, num_positions, margin_maintenance_flag, positions = process_swap_parallel(
                        manager,calculator, pair, order_size, data, positions, date, num_positions, effective_margin, margin_maintenance_flag, interval
                    )
                    effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)

                    margin_maintenance_flag, margin_maintenance_rate = update_margin_maintenance_rate(effective_margin, required_margin)
                    if margin_maintenance_flag:
                            print("executed loss cut in check swap")
                            #break
                    # ロスカット処理
                    effective_margin, margin_deposit, realized_profit, required_margin , margin_maintenance_flag, positions = process_losscut_parallel(
                        manager, margin_maintenance_flag, positions, price, effective_margin, margin_deposit, realized_profit, required_margin, order_size
                    )
                    if margin_maintenance_flag:
                            print("executed loss cut in losscut process")
                            #break
                    _, margin_maintenance_rate = update_margin_maintenance_rate(effective_margin,required_margin)  

                    # 有効証拠金の最大・最小値を確認
                    effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min)

            last_price = price
        # Calculate position value
        position_value = 0
        swap_value = 0
        print(positions)
        if positions:
            position_value += sum(size * (data.iloc[-1] - grid) if 'Buy' in status and not status.endswith('Closed') else
                         -size * (data.iloc[-1] - grid) if 'Sell'  in status and not status.endswith('Closed') else
                         0 for size, _, status, grid, _, _, _, _, _ in positions)
        else:
            position_value = 0
        # Calculate swap values  
        #"""
        if positions:
            swap_value += sum(calculator.get_total_swap_points(pair,status,data.index[index],date,size,data.index) if ('Buy' in status or 'Sell' in status) and not status.endswith('Closed') or 'Forced' in status else
                          0 for size, index, status, _, _, _, _, _, _ in positions) + sum(calculator.get_total_swap_points(pair,status,data.index[index],calculator.add_business_days(swap_day,1,data.index,interval,pair),size,data.index) if 'Closed' in status and calculator.add_business_days(swap_day,1,data.index,interval,pair) <= date and data.index[index] != swap_day else 0 for size, index, status, _, _, _, swap_day,_ ,_ in positions)
        else:
            swap_value = 0
        RETURN = (effective_margin, margin_deposit, position_value, swap_value, realized_profit, required_margin)

        print(f'最終有効証拠金: {RETURN[0]}, 預託証拠金: {RETURN[1]}, ポジション損益: {RETURN[2]} スワップ損益{RETURN[3]}, 実現利益: {RETURN[4]}, 必要証拠金: {RETURN[5]}')

        return RETURN


def check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min):
    effective_margin_max = max(effective_margin, effective_margin_max)
    effective_margin_min = min(effective_margin, effective_margin_min)
    return effective_margin_max, effective_margin_min


if __name__ == "__main__":
    pair = 'AUDNZD=X'
    interval="1d"
    website = "minkabu" #minkabu or  oanda
    end_date = datetime.strptime("2019-11-30","%Y-%m-%d")#datetime.now() - timedelta(days=7)
    #start_date = datetime.strptime("2019-09-01","%Y-%m-%d")#datetime.now() - timedelta(days=14)
    start_date = datetime.strptime("2019-11-01","%Y-%m-%d")#datetime.now() - timedelta(days=14)
    initial_funds = 100000000
    grid_start = 1.02
    grid_end = 1.14
    strategies = ['long_only']
    entry_intervals = [0]  # エントリー間隔
    total_thresholds = [10000]  # 全ポジション決済の閾値

    order_sizes = [1000]
    num_traps_options = [100]
    profit_widths = [0.01]
    densities = [10]
    data = fetch_currency_data(pair, start_date, end_date,interval)
    #data = data.copy()  # コピーを作成
    #data *= 100  # コピーを100倍

    calculator = SwapCalculator(website,pair,start_date,end_date,interval=interval)
    
    multiprocessing.set_start_method('spawn', force=True)

    for order_size, num_traps, profit_width, strategy, density, entry_interval, total_threshold in product(order_sizes, num_traps_options, profit_widths, strategies, densities, entry_intervals, total_thresholds):
        taple = traripi_backtest(calculator, data, initial_funds, grid_start, grid_end, num_traps, profit_width, order_size, entry_interval=None, total_threshold=None, strategy='standard', density=1)
        print(f"check total: {taple[1]+taple[2]+taple[3]}")

