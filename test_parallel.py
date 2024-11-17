import multiprocessing as mp
import numpy as np
from kabu_library import fetch_currency_data
from datetime import datetime
from kabu_swap import SwapCalculator

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

def process_grids_parallel(manager, date, grids, last_price, price, effective_margin, order_size, i, required_margin_rate, margin_maintenance_rate):
    # ロックオブジェクトを作成
    lock = manager.Lock()
    
    # 共有リソースを Manager で作成
    effective_margin = manager.Value('d', effective_margin)
    order_margin = manager.Value('d', 0.0)
    required_margin = manager.Value('d', 0.0)
    positions = manager.list()  # ポジションリスト
    order_capacity_flag = manager.Value('b', False)
    margin_maintenance_flag = manager.Value('b', False)
    
    # 並列処理用のプールを作成
    with mp.Pool() as pool:
        pool.starmap(process_grid, [(date, grid, last_price, price, effective_margin, positions, order_size, i, required_margin_rate, required_margin, margin_maintenance_rate, order_margin, order_capacity_flag, margin_maintenance_flag, lock) for grid in grids])
    
    return order_margin.value, required_margin.value, list(positions), order_capacity_flag.value, margin_maintenance_flag.value



def process_position(data, margin_maintenance_flag, effective_margin, margin_deposit, realized_profit, required_margin, pos, price, profit_width, order_size, required_margin_rate, lock):
    """各ポジションの状態を処理する関数"""
    if pos[2] == 'Buy' and pos[1] <= len(data) - 1:
        if price - pos[3] < profit_width:
            unrealized_profit = order_size * (price - pos[3])
            add_required_margin = -pos[5] + price * order_size * required_margin_rate
            with lock:
                effective_margin.value += unrealized_profit - pos[4]
                required_margin.value += add_required_margin
            pos[5] += add_required_margin
            pos[4] = unrealized_profit
            
            print(f"価格 {price} に対して有効証拠金を更新しました。現在の有効証拠金: {effective_margin.value}, unrealized_profit: {unrealized_profit}, pos[4]: {pos[4]}")

        if price - pos[3] >= profit_width:
            profit = order_size * profit_width
            add_effective_margin = profit - pos[4]
            with lock:
                effective_margin.value += add_effective_margin
                margin_deposit.value += profit
                realized_profit.value += profit
                pos[7] += profit
                required_margin.value -= pos[5]
                pos[5] = 0
                pos[2] = 'Buy-Closed'
            print(f"利益 {profit} で Buy ポジションをクローズしました。グリッド {pos[3]}, 有効証拠金: {effective_margin.value}, 必要証拠金: {required_margin.value} added_effective_margin: {add_effective_margin}, 確定利益: {realized_profit}")

        print(f"check effective_margin in def process_position: {effective_margin.value}")

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
    with mp.Pool() as pool:
        pool.starmap(process_position, [
            (data, margin_maintenance_flag, effective_margin, margin_deposit, realized_profit, required_margin, pos, price, profit_width, order_size, required_margin_rate, lock) 
            for pos in positions
        ])
    print(f"check effective_margin in def process_positions_parallel: {effective_margin.value}")
    
    return positions, effective_margin.value, margin_deposit.value, realized_profit.value, required_margin.value, margin_maintenance_flag.value

def process_swap(calculator, pair, order_size, data, pos, date, effective_margin, num_positions, lock):
    """スワップ処理を行う関数"""
    if pos[2] == "Buy" or (pos[2] == "Buy-Closed" and calculator.add_business_days(pos[6], 1, data.index) == date and data.index[pos[1]] != pos[6]):
        with lock:
            add_effective_margin = calculator.get_total_swap_points(pair, pos[2], pos[6], date, order_size, data.index)
            pos[8] += calculator.get_total_swap_points(pair,pos[2],pos[6],date,order_size,data.index)
            print(add_effective_margin)
            effective_margin.value += add_effective_margin
            num_positions.value += 1
            print(f'スワップが有効証拠金に追加されました: effective_margin: {effective_margin}, add_effective_margin: {add_effective_margin}, date: {date}, pos[6]: {pos[6]}')
        if "Closed" not in pos[2]:
            pos[6] = date

def process_swap_parallel(manager,calculator, pair, order_size, data, positions, date, num_positions, effective_margin, margin_maintenance_flag):
    """並列でスワップ処理を行う関数"""
    # ロックオブジェクトを作成
    lock = manager.Lock()
    
    positions = manager.list(positions)
    effective_margin = manager.Value('d', effective_margin)
    margin_maintenance_flag = manager.Value('b', False)
    num_positions = manager.Value('d', num_positions)

    with mp.Pool() as pool:
        pool.starmap(process_swap, [(calculator, pair, order_size, data, pos, date, effective_margin, num_positions, lock) for pos in positions])

    return effective_margin.value, num_positions.value, margin_maintenance_flag, positions

def main(calculator, i, last_price, price,initial_funds = 0, realized_profit=0, position_value=0, swap_value=0):
    # 初期値設定
    date = data.index[i]
    grid_start = 107
    grid_end = 109
    num_traps=100
    grids = np.linspace(grid_start, grid_end, num=num_traps)
    order_size = 1000
    required_margin_rate = 0.1
    margin_maintenance_rate = 150
    margin_deposit = initial_funds + realized_profit
    effective_margin = margin_deposit + position_value + swap_value
    profit_width = 0.0005
    website = "minkabu"
    num_positions = 0

    # Manager を生成し、共有リソースを作成
    with mp.Manager() as manager:
        #calculator = SwapCalculator(website,pair,start_date,end_date,interval=interval)
        # 並列処理を実行
        order_margin, required_margin, positions, order_capacity_flag, margin_maintenance_flag = process_grids_parallel(
            manager, date, grids, last_price, price, effective_margin, order_size, i, required_margin_rate, margin_maintenance_rate
        )
            # 並列処理を実行
        positions, effective_margin, margin_deposit, realized_profit, required_margin, margin_maintenance_flag = process_positions_parallel(
            manager, data, positions, effective_margin, margin_deposit, realized_profit, required_margin, margin_maintenance_flag, price, profit_width, order_size, required_margin_rate
        )
        # スワップ処理
        effective_margin, num_positions, margin_maintenance_flag, positions = process_swap_parallel(
            manager, calculator, pair, order_size, data, positions, date, num_positions, effective_margin, margin_maintenance_flag
        )

        # 結果を出力
        print("Final Effective Margin:", effective_margin)
        print("Final Margin Deposit:", margin_deposit)
        print("Final Realized Profit:", realized_profit)
        print("Final Required Margin:", required_margin)


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

        return effective_margin, margin_deposit, position_value, swap_value, realized_profit, required_margin

if __name__ == '__main__':
    pair = 'USDJPY=X'
    interval="1d"
    end_date = datetime.strptime("2021-11-15","%Y-%m-%d")#datetime.now() - timedelta(days=7)
    #start_date = datetime.strptime("2019-09-01","%Y-%m-%d")#datetime.now() - timedelta(days=14)
    start_date = datetime.strptime("2019-11-01","%Y-%m-%d")#datetime.now() - timedelta(days=14)
    mp.set_start_method('spawn', force=True)  # Mac 環境でのプロセス開始方法
    data = fetch_currency_data(pair, start_date, end_date,interval)

    #data *= 0.01
    website = "minkabu"
    calculator = SwapCalculator(website,pair,start_date,end_date,interval=interval)
    last_price = None
    realized_profit = 0
    initial_funds = 100000
    position_value = 0
    swap_value = 0
    for i in range(10):
        price = data.iloc[i]

        if last_price is not None:
            effective_margin, margin_deposit, position_value, swap_value, realized_profit, required_margin = main(calculator, i, last_price, price,initial_funds=initial_funds, realized_profit=realized_profit, position_value=position_value, swap_value=swap_value)

        last_price = price
