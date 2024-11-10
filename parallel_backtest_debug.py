import numpy as np
from multiprocessing import Pool, Manager, Lock
from itertools import product
from kabu_swap import SwapCalculator
from kabu_library import fetch_currency_data
from datetime import datetime
import pandas as pd

def process_grid(date, grid, last_price, price, effective_margin, positions, order_size, i, required_margin_rate, required_margin, margin_maintenance_rate, order_margin, order_capacity_flag, margin_maintenance_flag, lock):
    """グリッドでの取引処理を行う関数"""
    if min(last_price, price) <= grid <= max(last_price, price):
        if margin_maintenance_rate <= 100:
            margin_maintenance_flag = True
            return
        order_capacity = effective_margin - (required_margin + order_margin)
        if order_capacity < 0:
            order_capacity_flag = True
            print('発注容量が不足しているため、発注できません。')
            return
        if margin_maintenance_rate > 100 and order_capacity > 0:
            subtract_order_margin = order_size * grid * required_margin_rate
            add_required_margin = grid * order_size * required_margin_rate
            with lock:
                order_margin -= subtract_order_margin
                required_margin += add_required_margin
                positions.append([order_size, i, 'Buy', grid, 0, add_required_margin, date, 0, 0])
            print(f"グリッド {grid} で Buy ポジションをオープンしました。")

def process_grids_parallel(positions, order_margin, required_margin, date, grids, last_price, price, effective_margin, order_size, i, required_margin_rate, margin_maintenance_rate, order_capacity_flag, margin_maintenance_flag, lock):
    """並列でグリッド処理を行う関数"""
    with Pool() as pool:
        pool.starmap(process_grid, [(date, grid, last_price, price, effective_margin ,positions, order_size, i, required_margin_rate, required_margin, margin_maintenance_rate, order_margin, order_capacity_flag, margin_maintenance_flag, lock) for grid in grids])
    
    return order_margin, required_margin, positions, order_capacity_flag, margin_maintenance_flag

def process_position(data, margin_maintenance_flag, effective_margin, margin_deposit, realized_profit, required_margin, pos, price, profit_width, order_size, required_margin_rate, lock):
    """各ポジションの状態を処理する関数"""
    if pos[2] == 'Buy' and pos[1] <= len(data) - 1:
        if price - pos[3] < profit_width:
            unrealized_profit = order_size * (price - pos[3])
            add_required_margin = -pos[5] + price * order_size * required_margin_rate
            with lock:
                effective_margin += unrealized_profit - pos[4]
                required_margin += add_required_margin
            pos[5] += add_required_margin
            pos[4] = unrealized_profit
            
            print(f"価格 {price} に対して有効証拠金を更新しました。現在の有効証拠金: {effective_margin}")

        if price - pos[3] >= profit_width:
            profit = order_size * profit_width
            add_effective_margin = profit - pos[4]
            with lock:
                effective_margin += add_effective_margin
                margin_deposit += profit
                realized_profit += profit
                pos[7] += profit
                required_margin -= pos[5]
                pos[5] = 0
                pos[2] = 'Buy-Closed'
            print(f"利益 {profit} で Buy ポジションをクローズしました。グリッド {pos[3]}, 有効証拠金: {effective_margin}, 必要証拠金: {required_margin}")

def process_positions_parallel(data, positions, effective_margin, margin_deposit, realized_profit, required_margin, margin_maintenance_flag, price, profit_width, order_size, required_margin_rate, lock):
    """並列でポジション処理を行う関数"""
    with Pool() as pool:
        pool.starmap(process_position, [(data,margin_maintenance_flag, effective_margin, margin_deposit, realized_profit, required_margin, pos, price, profit_width, order_size, required_margin_rate, lock) for pos in positions])

    return positions, effective_margin, margin_deposit, realized_profit, required_margin, margin_maintenance_flag

def process_swap(calculator, pair, order_size, data, pos, date, effective_margin, num_positions, lock):
    """スワップ処理を行う関数"""
    if pos[2] == "Buy" or (pos[2] == "Buy-Closed" and calculator.add_business_days(pos[6], 1, data.index) == date and data.index[pos[1]] != pos[6]):
        add_effective_margin = calculator.get_total_swap_points(pair, pos[2], pos[6], date, order_size, data.index)
        with lock:
            effective_margin += add_effective_margin
            num_positions += 1
        print(f'スワップが有効証拠金に追加されました: {effective_margin}')
        if "Closed" not in pos[2]:
            pos[6] = date

def process_swap_parallel(calculator, pair, order_size, data, positions, date, num_positions, effective_margin, margin_maintenance_flag, lock):
    """並列でスワップ処理を行う関数"""
    with Pool() as pool:
        pool.starmap(process_swap, [(calculator, pair, order_size, data, pos, date, effective_margin, num_positions, lock) for pos in positions])

    return effective_margin, num_positions, margin_maintenance_flag

def process_losscut(margin_maintenance_flag, pos, price, effective_margin, margin_deposit, realized_profit, required_margin, lock):
    """ロスカット処理を行う関数"""
    if margin_maintenance_flag:
        if pos[2] == 'Buy':
            profit = (price - pos[3]) * order_size
            with lock:
                effective_margin.value += profit - pos[4]
                margin_deposit.value += profit
                realized_profit.value += profit
                required_margin.value -= pos[5]
            pos[7] += profit
            pos[5] = 0
            pos[2] = "Buy-Closed"
            print(f"価格 {price} で強制クローズしました。グリッド {pos[3]}, 有効証拠金: {effective_margin}")

def process_losscut_parallel(margin_maintenance_flag, positions, price, effective_margin, margin_deposit, realized_profit, required_margin, lock):
    """並列でロスカット処理を行う関数"""
    with Pool() as pool:
        pool.starmap(process_losscut, [(margin_maintenance_flag, pos, price, effective_margin, margin_deposit, realized_profit, required_margin, lock) for pos in positions])

    return effective_margin, margin_deposit, realized_profit, required_margin, margin_maintenance_flag


def update_margin_maintenance_rate(effective_margin, required_margin):
    if required_margin != 0:  
        margin_maintenance_rate = (effective_margin / required_margin) * 100
    else:
        margin_maintenance_rate = np.inf

    if margin_maintenance_rate <= 100:
        print(f"Margin maintenance rate is {margin_maintenance_rate}%, below threshold. Forced liquidation triggered.")
        return True, margin_maintenance_rate  # フラグと値を返す
    return False, margin_maintenance_rate  # フラグと値を返す


def traripi_backtest(calculator, data, initial_funds, grid_start, grid_end, num_traps, profit_width, order_size, entry_interval=None, total_threshold=None, strategy='standard', density=1):
    """トラリピバックテストを実行する関数"""
    RETURN = []
    effective_margin_max = -np.inf
    effective_margin_min = np.inf
    margin_maintenance_rate = float('inf')
    required_margin_rate = 0.04

    with Manager() as manager:
        margin_deposit = manager.Value('d', initial_funds)
        realized_profit = manager.Value('d', 0)
        order_capacity_flag = manager.Value('b', False)
        positions = manager.list()
        order_margin = manager.Value('d', sum(order_size * grid * required_margin_rate for grid in np.linspace(grid_start, grid_end, num=num_traps)))
        required_margin = manager.Value('d', 0)
        effective_margin = manager.Value('d', margin_deposit.value)
        margin_maintenance_flag = manager.Value('b', False)
        num_positions = manager.Value('d', 0)
        lock = manager.Lock()

        last_price = None

        for i in range(len(data)):
            if margin_maintenance_flag.value or order_capacity_flag.value:
                break
            date = data.index[i]
            price = data.iloc[i]

            if last_price is not None:
                if price != last_price:
                    margin_maintenance_flag.value, margin_maintenance_rate = update_margin_maintenance_rate(effective_margin.value,required_margin.value)
                    if margin_maintenance_flag.value:
                        break

                    order_capacity = effective_margin.value - (required_margin.value + order_margin.value)
                    if order_capacity < 0:
                        order_capacity_flag.value = True
                        print(f'cannot order because of lack of order capacity')
                        break

                    if margin_maintenance_rate > 100 and order_capacity > 0:

                        order_margin.value, required_margin.value, positions, order_capacity_flag.value, margin_maintenance_flag.value = process_grids_parallel(
                            positions, order_margin.value, required_margin.value, date, np.linspace(grid_start, grid_end, num=num_traps), last_price, price, effective_margin.value, order_size, i, required_margin_rate, margin_maintenance_rate, order_capacity_flag.value, margin_maintenance_flag.value, lock
                        )

                        margin_maintenance_flag.value, margin_maintenance_rate = update_margin_maintenance_rate(effective_margin.value, required_margin.value)
                        if margin_maintenance_flag.value:
                            print("executed loss cut in last_price <= grid < price")
                            break                            

            # ポジション処理
            positions, effective_margin.value, margin_deposit.value, realized_profit.value, required_margin.value, margin_maintenance_flag.value = process_positions_parallel(
                data, positions, effective_margin.value, margin_deposit.value, realized_profit.value, required_margin.value, margin_maintenance_flag.value, price, profit_width, order_size, required_margin_rate, lock
            )

            effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin.value, effective_margin_max, effective_margin_min)
            
            margin_maintenance_flag.value, margin_maintenance_rate = update_margin_maintenance_rate(effective_margin.value, required_margin.value)
            if margin_maintenance_flag.value:
                    print("executed loss cut in position closure")
                    break



            last_price = price


            # スワップ処理
            effective_margin.value, num_positions.value, margin_maintenance_flag.value = process_swap_parallel(
                calculator, pair, order_size, data, positions, date, num_positions.value, effective_margin.value, margin_maintenance_flag.value, lock
            )

            

            margin_maintenance_flag.value, margin_maintenance_rate = update_margin_maintenance_rate(effective_margin.value, required_margin.value)
            if margin_maintenance_flag.value:
                    print("executed loss cut in check swap")
                    break


            # ロスカット処理
            effective_margin.value, margin_deposit.value, realized_profit.value, required_margin.value, margin_maintenance_flag.value = process_losscut_parallel(
                margin_maintenance_flag.value, positions, price, effective_margin.value, margin_deposit.value, realized_profit.value, required_margin.value, lock
            )
            _, margin_maintenance_rate = update_margin_maintenance_rate(effective_margin.value,required_margin.value)  

            
            # 有効証拠金の最大・最小値を確認
            effective_margin_max, effective_margin_min = check_min_max_effective_margin(effective_margin.value, effective_margin_max, effective_margin_min)

        RETURN = (effective_margin.value, margin_deposit.value, realized_profit.value, required_margin.value)
        print(f'最終有効証拠金: {RETURN[0]}, マージンデポジット: {RETURN[1]}, 実現利益: {RETURN[2]}, 必要証拠金: {RETURN[3]}')

    return RETURN


def check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min):
    effective_margin_max = max(effective_margin, effective_margin_max)
    effective_margin_min = min(effective_margin, effective_margin_min)
    return effective_margin_max, effective_margin_min


if __name__ == "__main__":
    pair = 'AUDNZD=X'
    interval="1d"
    website = "minkabu" #minkabu or  oanda
    end_date = datetime.strptime("2021-01-15","%Y-%m-%d")#datetime.now() - timedelta(days=7)
    #start_date = datetime.strptime("2019-09-01","%Y-%m-%d")#datetime.now() - timedelta(days=14)
    start_date = datetime.strptime("2019-11-12","%Y-%m-%d")#datetime.now() - timedelta(days=14)
    initial_funds = 10000000
    grid_start = 1.02
    grid_end = 1.14
    strategies = ['long_only']
    entry_intervals = [0]  # エントリー間隔
    total_thresholds = [10000]  # 全ポジション決済の閾値

    order_sizes = [3000]
    num_traps_options = [100]
    profit_widths = [100]
    densities = [10]

    data = fetch_currency_data(pair, start_date, end_date,interval)
    calculator =  SwapCalculator(website,pair,start_date,end_date,interval=interval)
    for order_size, num_traps, profit_width, strategy, density, entry_interval, total_threshold in product(order_sizes, num_traps_options, profit_widths, strategies, densities, entry_intervals, total_thresholds):
        taple = traripi_backtest(calculator, data, initial_funds, grid_start, grid_end, num_traps, profit_width, order_size, entry_interval=None, total_threshold=None, strategy='standard', density=1)

