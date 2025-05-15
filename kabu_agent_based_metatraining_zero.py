import numpy as np
import math
import os
import random
from collections import OrderedDict
from common.layers2 import *
import json


    

seed = 0
initial_scale_factor = Variable(1.0)

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)  # Python の乱数シード
    np.random.seed(seed)  # NumPy の乱数シード

def combine_variables(vars):
    values = np.array([v.value for v in vars])

    def make_local_grad_fn(i):
        def local_grad_fn(grad):
            return grad[i]  # grad is vector, return scalar component
        return local_grad_fn

    parents = [(v, make_local_grad_fn(i)) for i, v in enumerate(vars)]
    return Variable(values, parents=parents)


def split_vector(x):
    return [Variable(x.value[i], parents=[(x, lambda g, i=i: g[i])]) for i in range(len(x.value))]

class MarketGenerator:
    def __init__(self, input_size, hidden_size, output_size, scale_factor=initial_scale_factor):
        # ランダム初期化（正規分布）
        self.params = OrderedDict()
        self.params['W1'] = Variable(np.random.randn(input_size, hidden_size) * 0.01)
        self.params['b1'] = Variable(0.0)
        self.params['W2'] = Variable(np.random.randn(hidden_size, output_size) * 0.01)
        self.params['b2'] = Variable(0.0)
        self.params['scale_factor'] = scale_factor

        self.layers = OrderedDict()
        self.layers['Affine1'] = lambda x: affine(x, self.params['W1'], self.params['b1'])
        self.layers['Sigmoid1'] = sigmoid
        self.layers['Affine2'] = lambda x: affine(x, self.params['W2'], self.params['b2'])
        self.layers['Sigmoid2'] = softplus
        self.layers['LogScale'] = lambda x: mul(x, self.params['scale_factor'])

        self.optimizer = Adam()

    def predict(self, x):
        for name, layer in self.layers.items():
            x = layer(x)
        return x

    def gradient(self, loss):
        loss.backward()
        grads = {
            name: loss.grad(param)
            for name, param in self.params.items()
        }
        return grads
    
    def numerical_gradient(self, x, loss_func, h=1e-4):
        grads = {}
        for name, param in self.params.items():
            grad = np.zeros_like(param.value)
            original_value = param.value.copy()

            # 多次元対応：各要素にhを加減して数値微分
            it = np.nditer(param.value, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                tmp = param.value[idx]

                param.value[idx] = tmp + h
                fxh1 = loss_func(self.predict(x)).value

                param.value[idx] = tmp - h
                fxh2 = loss_func(self.predict(x)).value

                grad[idx] = (fxh1 - fxh2) / (2 * h)
                param.value[idx] = tmp
                it.iternext()

            grads[name] = grad
            param.value = original_value
        return grads



"""
def validate_unfulfilled_orders(agents, previous_unfulfilled_long_open, previous_unfulfilled_short_open, 
                            previous_unfulfilled_long_close, previous_unfulfilled_short_close,
                            final_remaining_long_open, final_remaining_short_open, 
                            final_remaining_long_close, final_remaining_short_close):

    #各エージェントの unfulfilled (未約定) の新規注文・決済注文が整合性を持っているかをチェックする関数。


    # 🔹 3️⃣ 各エージェントの現在の未約定注文の合計を取得
    total_actual_unfulfilled_long_open = sum_variables(agent.unfulfilled_long_open for agent in agents) - previous_unfulfilled_long_open
    total_actual_unfulfilled_short_open = sum_variables(agent.unfulfilled_short_open for agent in agents) - previous_unfulfilled_short_open
    total_actual_unfulfilled_long_close = sum_variables(agent.unfulfilled_long_close for agent in agents) - previous_unfulfilled_long_close
    total_actual_unfulfilled_short_close = sum_variables(agent.unfulfilled_short_close for agent in agents) - previous_unfulfilled_short_close

    # 🔹 4️⃣ 整合性チェック
    print(f"Expected Unfulfilled Long Open: {final_remaining_long_open}, Actual: {total_actual_unfulfilled_long_open}")
    print(f"Expected Unfulfilled Short Open: {final_remaining_short_open}, Actual: {total_actual_unfulfilled_short_open}")
    print(f"Expected Unfulfilled Long Close: {final_remaining_long_close}, Actual: {total_actual_unfulfilled_long_close}")
    print(f"Expected Unfulfilled Short Close: {final_remaining_short_close}, Actual: {total_actual_unfulfilled_short_close}")

    assert abs(final_remaining_long_open - total_actual_unfulfilled_long_open) < 1e-6, \
        f"❌ Mismatch in unfulfilled long open orders! Expected: {final_remaining_long_open}, Got: {total_actual_unfulfilled_long_open}"

    assert abs(final_remaining_short_open - total_actual_unfulfilled_short_open) < 1e-6, \
        f"❌ Mismatch in unfulfilled short open orders! Expected: {final_remaining_short_open}, Got: {total_actual_unfulfilled_short_open}"

    assert abs(final_remaining_long_close - total_actual_unfulfilled_long_close) < 1e-6, \
        f"❌ Mismatch in unfulfilled long close orders! Expected: {final_remaining_long_close}, Got: {total_actual_unfulfilled_long_close}"

    assert abs(final_remaining_short_close - total_actual_unfulfilled_short_close) < 1e-6, \
        f"❌ Mismatch in unfulfilled short close orders! Expected: {final_remaining_short_close}, Got: {total_actual_unfulfilled_short_close}"

    print("✅ Unfulfilled order validation passed successfully!")
"""



def check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min):
    if effective_margin_max < effective_margin:
        effective_margin_max = effective_margin
    if effective_margin_min > effective_margin:
        effective_margin_min = effective_margin
    return effective_margin_max, effective_margin_min


def update_margin_maintenance_rate(effective_margin, required_margin, margin_cut_threshold=100.0):
    if required_margin != 0:  
        margin_maintenance_rate = (effective_margin / required_margin) * 100
    else:
        margin_maintenance_rate = np.inf

    if margin_maintenance_rate <= margin_cut_threshold:
        print(f"Margin maintenance rate is {margin_maintenance_rate}%, below threshold. Forced losscut triggered.")
        return True, margin_maintenance_rate  # フラグと値を返す
    return False, margin_maintenance_rate  # フラグと値を返す

class RLAgent():
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01, initial_cash=100000):
        self.positions = []
        self.closed_positions = []
        self.effective_margin = Variable(initial_cash)
        self.required_margin = 0
        self.margin_deposit = Variable(initial_cash)
        self.realized_profit = Variable(0.0)
        self.long_position = Variable(0.0)
        self.short_position = Variable(0.0)
        self.unfulfilled_long_open = Variable(0.0)
        self.unfulfilled_short_open = Variable(0.0)
        self.unfulfilled_long_close = Variable(0.0)
        self.unfulfilled_short_close = Variable(0.0)
        self.margin_maintenance_rate = np.inf
        self.margin_maintenance_flag = False
        self.effective_margin_max = -np.inf
        self.effective_margin_min = np.inf

        self.params = OrderedDict()
        self.params['W1'] = Variable(np.random.randn(input_size, hidden_size) * 0.01)
        self.params['b1'] = Variable(0.0)
        self.params['W2'] = Variable(np.random.randn(hidden_size, output_size) * 0.01)
        self.params['b2'] = Variable(0.0)

        self.layers = OrderedDict()
        self.layers['Affine1'] = lambda x: affine(x, self.params['W1'], self.params['b1'])
        self.layers['Sigmoid1'] = sigmoid
        self.layers['Affine2'] = lambda x: affine(x, self.params['W2'], self.params['b2'])
        self.layers['Sigmoid2'] = relu

        self.optimizer = Adam()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x

    def gradient(self, loss):
        loss.backward()
        grads = {
            name: loss.grad(param)
            for name, param in self.params.items()
        }
        return grads

    def numerical_gradient(self, x, loss_func, h=1e-4):
        grads = {}
        for name, param in self.params.items():
            grad = np.zeros_like(param.value)
            original_value = param.value.copy()

            # 多次元対応：各要素にhを加減して数値微分
            it = np.nditer(param.value, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                tmp = param.value[idx]

                param.value[idx] = tmp + h
                fxh1 = loss_func(self.predict(x)).value

                param.value[idx] = tmp - h
                fxh2 = loss_func(self.predict(x)).value

                grad[idx] = (fxh1 - fxh2) / (2 * h)
                param.value[idx] = tmp
                it.iternext()

            grads[name] = grad
            param.value = original_value
        return grads

    

    def process_new_order(self, long_order_size, short_order_size, current_price, margin_rate):
        if long_order_size.value > 0 and short_order_size.value > 0:
            order_volume = add(long_order_size, short_order_size)
            total_open = add(order_volume, add(self.unfulfilled_short_open, self.unfulfilled_long_open))
            order_margin = mul(mul(total_open, current_price), margin_rate)

            if self.margin_maintenance_flag:
                print(f"Margin cut triggered during order processing.")
                return

            margin_total = add(self.required_margin, order_margin)
            order_capacity = sub(self.effective_margin, margin_total)

            if order_capacity.value < 0:
                self.unfulfilled_long_open = add(self.unfulfilled_long_open, long_order_size)
                self.unfulfilled_short_open = add(self.unfulfilled_short_open, short_order_size)
                return

            if self.margin_maintenance_rate > 100 and order_capacity.value > 0:
                long_add_required_margin = mul(mul(add(long_order_size, self.unfulfilled_long_open), current_price), margin_rate)
                short_add_required_margin = mul(mul(add(short_order_size, self.unfulfilled_short_open), current_price), margin_rate)
                self.required_margin = add(self.required_margin, add(long_add_required_margin, short_add_required_margin))

                if long_order_size.value > 0:
                    long_order_size = add(long_order_size, self.unfulfilled_long_open)
                    self.unfulfilled_long_open = Variable(0.0)
                    pos = [long_order_size, Variable(1.0), current_price, Variable(0.0), long_add_required_margin, Variable(0.0)]
                    print(f"Opened Buy position at {current_price.value}, required_margin:{self.required_margin.value}")
                    self.positions.append(pos)

                if short_order_size.value > 0:
                    short_order_size = add(short_order_size, self.unfulfilled_short_open)
                    self.unfulfilled_short_open = Variable(0.0)
                    pos = [short_order_size, Variable(-1.0), current_price, Variable(0.0), short_add_required_margin, Variable(0.0)]
                    print(f"Opened Sell position at {current_price.value}, required_margin:{self.required_margin.value}")
                    self.positions.append(pos)

                print(f"positions in process_new_order: {len(self.positions)}")
                self.margin_maintenance_flag, self.margin_maintenance_rate = update_margin_maintenance_rate(self.effective_margin, self.required_margin)
                if self.margin_maintenance_flag:
                    print(f"margin maintenance rate is {self.margin_maintenance_rate}, so loss cut is executed in process_new_order, effective_margin: {self.effective_margin.value}")
                    return



    def process_position_closure(self, long_close_position, short_close_position, current_price):
        if self.margin_maintenance_flag:
            print("Margin cut triggered during position closure.")
            return

        self.unfulfilled_long_close = add(self.unfulfilled_long_close, long_close_position)
        self.unfulfilled_short_close = add(self.unfulfilled_short_close, short_close_position)

        to_be_removed = []
        for pos_id in range(len(self.positions)):
            pos = self.positions[pos_id]
            size, pos_type, open_price, unrealized_profit, margin, realized_profit = pos

            if pos_type.value == 1.0 and self.unfulfilled_long_close.value > 0:
                fulfilled_size = min_var(self.unfulfilled_long_close.value, size)
                profit = mul(fulfilled_size, sub(current_price, open_price))
            elif pos_type.value == -1.0 and self.unfulfilled_short_close.value > 0:
                fulfilled_size = min_var(self.unfulfilled_short_close.value, size)
                profit = mul(fulfilled_size, sub(open_price, current_price))
            else:
                continue

            self.effective_margin = add(self.effective_margin, sub(profit, unrealized_profit))
            self.effective_margin_max, self.effective_margin_min = check_min_max_effective_margin(self.effective_margin, self.effective_margin_max, self.effective_margin_min)
            
            self.margin_deposit = add(self.margin_deposit, profit)
            self.realized_profit = add(self.realized_profit, profit)

            ratio = div(fulfilled_size, size)
            add_required_margin = mul(margin, sub(Variable(0.0), ratio))
            self.required_margin += add_required_margin.value

            size -= fulfilled_size
            if pos_type.value == 1.0:
                self.unfulfilled_long_close = sub(self.unfulfilled_long_close, fulfilled_size)
            if pos_type.value == -1.0:
                self.unfulfilled_short_close = sub(self.unfulfilled_short_close, fulfilled_size)

            if size > 0:
                updated_margin = mul(margin, div(size, size + fulfilled_size))
                self.positions[pos_id] = [size, pos_type, open_price, Variable(0.0), updated_margin, Variable(0.0)]
                self.closed_positions.append([fulfilled_size, pos_type, open_price, Variable(0.0), Variable(0.0), profit])
            else:
                self.closed_positions.append([fulfilled_size, pos_type, open_price, Variable(0.0), Variable(0.0), profit])
                to_be_removed.append(pos_id)

            #    self.unfulfilled_short_open = short_close_position - fulfilled_size

            self.margin_maintenance_flag, self.margin_maintenance_rate = update_margin_maintenance_rate(self.effective_margin,self.required_margin)
            if self.margin_maintenance_flag:
                print(f"margin maintenance rate is {self.margin_maintenance_rate},so loss cut is executed in position closure process, effective_margin: {self.effective_margin}")
                continue
        for pos_id in sorted(to_be_removed, reverse=True):  # 降順で削除（pos_id がズレないように）
            del self.positions[pos_id]


    def process_position_update(self, current_price, required_margin_rate):
        if self.margin_maintenance_flag:
            print(f"Margin cut triggered during position update.")
            return

        pos_id_max = len(self.positions) - 1
        for pos_id in range(pos_id_max + 1):
            try:
                pos = self.positions[pos_id]
                size, pos_type, open_price, before_unrealized_profit, margin, _ = pos
            except:
                continue

            if pos_type.value == 1:
                unrealized_profit = mul(size, sub(current_price, open_price))
            else:
                unrealized_profit = mul(size, sub(open_price, current_price))

            self.effective_margin = add(self.effective_margin, sub(unrealized_profit, before_unrealized_profit))
            self.effective_margin_max, self.effective_margin_min = check_min_max_effective_margin(
                self.effective_margin, self.effective_margin_max, self.effective_margin_min
            )

            new_required_margin = sub(mul(size, mul(current_price, required_margin_rate)), margin)
            self.required_margin += new_required_margin

            pos = [size, pos_type, open_price, unrealized_profit, new_required_margin, 0]
            self.positions[pos_id] = pos

            self.margin_maintenance_flag, self.margin_maintenance_rate = update_margin_maintenance_rate(
                self.effective_margin, self.required_margin
            )
            if self.margin_maintenance_flag:
                print(f"margin maintenance rate is {self.margin_maintenance_rate}, so loss cut is executed.")

        self.margin_maintenance_flag, _ = update_margin_maintenance_rate(
            self.effective_margin, self.required_margin
        )
        if self.margin_maintenance_flag:
            print("Forced margin cut triggered.")
            to_be_removed = []
            for pos_id in range(len(self.positions)):
                try:
                    pos = self.positions[pos_id]
                    size, pos_type, open_price, before_unrealized_profit, margin, _ = pos
                except:
                    continue

                if pos_type.value == 1:
                    profit = mul(size, sub(current_price, open_price))
                else:
                    profit = mul(size, sub(open_price, current_price))

                self.effective_margin = add(self.effective_margin, sub(profit, before_unrealized_profit))
                self.effective_margin_max, self.effective_margin_min = check_min_max_effective_margin(
                    self.effective_margin, self.effective_margin_max, self.effective_margin_min
                )
                self.margin_deposit = add(self.margin_deposit, profit)
                self.realized_profit = add(self.realized_profit, profit)
                self.required_margin -= margin

                pos = [size, pos_type, open_price, 0, 0, profit]
                self.closed_positions.append(pos)
                to_be_removed.append(pos_id)

            for pos_id in sorted(to_be_removed, reverse=True):
                del self.positions[pos_id]


            self.required_margin = 0
            _, self.margin_maintenance_rate = update_margin_maintenance_rate(
                self.effective_margin, self.required_margin
            )


        #"""


def match_orders(agents, actions, current_price, required_margin_rate):
    # 1️⃣ 各エージェントの注文を取得
    long_open_orders = []
    short_open_orders = []
    long_close_orders = []
    short_close_orders = []

    for agent, action in zip(agents, actions):
        long_open, short_open, long_close, short_close = split_vector(action)
        long_open_orders.append(long_open)
        short_open_orders.append(short_open)
        long_close_orders.append(long_close)
        short_close_orders.append(short_close)

    # 2️⃣ 需給に基づいたマッチング処理
    total_long_open = sum_variables(long_open_orders)
    total_short_open = sum_variables(short_open_orders)
    total_long_close = sum_variables(long_close_orders)
    total_short_close = sum_variables(short_close_orders)

    executed_open_volume = min_var(total_long_open, total_short_open)
    executed_close_volume = min_var(total_long_close, total_short_close)

    # 3️⃣ 各エージェントへの実行割当
    for i, (agent, long_open, short_open, long_close, short_close) in enumerate(zip(
        agents, long_open_orders, short_open_orders, long_close_orders, short_close_orders)):

        # 分母が0のときは0で割らないように制御
        long_open_ratio = div(long_open, total_long_open) if total_long_open.value > 0 else Variable(0.0)
        short_open_ratio = div(short_open, total_short_open) if total_short_open.value > 0 else Variable(0.0)
        long_close_ratio = div(long_close, total_long_close) if total_long_close.value > 0 else Variable(0.0)
        short_close_ratio = div(short_close, total_short_close) if total_short_close.value > 0 else Variable(0.0)

        executed_long_open = mul(executed_open_volume, long_open_ratio)
        executed_short_open = mul(executed_open_volume, short_open_ratio)
        executed_long_close = mul(executed_close_volume, long_close_ratio)
        executed_short_close = mul(executed_close_volume, short_close_ratio)

        print(f"\nprocess_new_order of {i}th agent by ordering new positions")
        agent.process_new_order(executed_long_open, executed_short_open, current_price, required_margin_rate)
        print(f"\nprocess_position_closure of {i}th agent by closing positions")
        agent.process_position_closure(executed_long_close, executed_short_close, current_price)

    # 4️⃣ 残り注文の相殺
    remaining_long_open = sub(total_long_open, executed_open_volume)
    remaining_short_open = sub(total_short_open, executed_open_volume)
    remaining_long_close = sub(total_long_close, executed_close_volume)
    remaining_short_close = sub(total_short_close, executed_close_volume)

    total_buy_side = add(remaining_long_open, remaining_short_close)
    total_sell_side = add(remaining_short_open, remaining_long_close)

    executed_cross_volume = min_var(total_buy_side, total_sell_side)

    if executed_cross_volume.value > 0:
        executed_longs = mul(executed_cross_volume, div(remaining_long_open, total_buy_side)) if total_buy_side.value > 0 else Variable(0.0)
        executed_shorts = mul(executed_cross_volume, div(remaining_short_close, total_buy_side)) if total_buy_side.value > 0 else Variable(0.0)
        executed_sells = mul(executed_cross_volume, div(remaining_short_open, total_sell_side)) if total_sell_side.value > 0 else Variable(0.0)
        executed_buys = mul(executed_cross_volume, div(remaining_long_close, total_sell_side)) if total_sell_side.value > 0 else Variable(0.0)

        final_remaining_long_open = sub(remaining_long_open, executed_longs)
        final_remaining_short_open = sub(remaining_short_open, executed_shorts)
        final_remaining_long_close = sub(remaining_long_close, executed_buys)
        final_remaining_short_close = sub(remaining_short_close, executed_sells)

        for i, (agent, long_open, short_open, long_close, short_close) in enumerate(zip(
            agents, long_open_orders, short_open_orders, long_close_orders, short_close_orders)):

            long_open_ratio = div(long_open, total_long_open) if total_long_open.value > 0 else Variable(0.0)
            short_open_ratio = div(short_open, total_short_open) if total_short_open.value > 0 else Variable(0.0)
            long_close_ratio = div(long_close, total_long_close) if total_long_close.value > 0 else Variable(0.0)
            short_close_ratio = div(short_close, total_short_close) if total_short_close.value > 0 else Variable(0.0)

            executed_long_open = mul(executed_longs, long_open_ratio)
            executed_short_open = mul(executed_shorts, short_open_ratio)
            executed_long_close = mul(executed_buys, long_close_ratio)
            executed_short_close = mul(executed_sells, short_close_ratio)

            print(f"\nprocess_new_order of {i}th agent by offsetting remaining orders")
            agent.process_new_order(executed_long_open, executed_short_open, current_price, required_margin_rate)
            print(f"\nprocess_position_closure of {i}th agent by offsetting remaining orders")
            agent.process_position_closure(executed_long_close, executed_short_close, current_price)

            # 未約定の更新（Variableに加算）
            agent.unfulfilled_long_open = add(agent.unfulfilled_long_open, mul(final_remaining_long_open, long_open_ratio))
            agent.unfulfilled_short_open = add(agent.unfulfilled_short_open, mul(final_remaining_short_open, short_open_ratio))
            agent.unfulfilled_long_close = add(agent.unfulfilled_long_close, mul(final_remaining_long_close, long_close_ratio))
            agent.unfulfilled_short_close = add(agent.unfulfilled_short_close, mul(final_remaining_short_close, short_close_ratio))

    print(f"executed_open_volume: {executed_open_volume.value}")
    print(f"executed_close_volume: {executed_close_volume.value}")


def serialize(obj):
    if isinstance(obj, Variable):
        val = obj.value
        return val.tolist() if isinstance(val, np.ndarray) else val

    elif isinstance(obj, np.ndarray):
        return obj.tolist()

    elif isinstance(obj, list):
        return [serialize(v) for v in obj]

    elif isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}

    else:
        return obj


if __name__ == "__main__":
    num_agents = 5
    use_rule_based = True
    generations = 165
    required_margin_rate = 0.04
    gamma = Variable(1.0)
    volume = Variable(0)

    set_seed(0)
    agents = [RLAgent(input_size=2,hidden_size=128,output_size=4) for _ in range(num_agents)]
    generator = MarketGenerator(input_size=4,hidden_size=128,output_size=3)

    states = [Variable(100.0), Variable(1.0), Variable(0.01)]
    supply_and_demand = Variable(0.0)

    history = {
        "generated_states": [],
        "actions": [],
        "agent_assets": [],
        "liquidity": [],
        "slippage": [],
        "gen_gradients": [],
        "disc_gradients": [],
        "gen_loss": [],
        "disc_losses": [],
        "scale_factor": [],
    }

    for generation in range(generations):
        set_seed(generation)

        log_inputs = [
            log(add(s, Variable(1e-6))) for s in states
        ]
        log_supply = log(add(abs_var(supply_and_demand), Variable(1e-6)))
        signed_log_supply = mul(sign(supply_and_demand), log_supply)
        log_inputs.append(signed_log_supply)

        log_inputs_vec = combine_variables(log_inputs)

        generated_states = generator.predict(log_inputs_vec)

        price_var, liquidity_var, slippage_var = split_vector(generated_states)
        current_price = sub(exp(price_var), Variable(1e-6))
        current_liquidity = sub(exp(liquidity_var), Variable(1e-6))
        current_slippage = sub(exp(slippage_var), Variable(1e-6))

        if use_rule_based:
            k = div(Variable(1.0),add(Variable(1.0),mul(gamma,volume)))
            current_liquidity = div(Variable(1.0), add(Variable(1.0), mul(k, abs_var(supply_and_demand))))
            current_slippage = div(abs_var(supply_and_demand), add(current_liquidity, Variable(1e-6)))

        states = [current_price, current_liquidity, current_slippage]

        actions = []
        for agent in agents:
            log_inputs = [
                log(add(agent.effective_margin, Variable(1e-6))),
                log(add(current_price, Variable(1e-6)))
            ]

            log_inputs_vec = combine_variables(log_inputs)

            log_action = agent.predict(log_inputs_vec)
            action = sub(exp(log_action),Variable(1e-6))
            actions.append(action)

        volume = sum_variables([abs_var(a) for a in actions])

        match_orders(agents, actions, current_price, Variable(required_margin_rate))

        supply_and_demand = sum_variables([
            sub(add(agent.unfulfilled_long_open, agent.unfulfilled_short_close),
                add(agent.unfulfilled_short_open, agent.unfulfilled_long_close))
            for agent in agents
        ])

        for agent in agents:
            agent.process_position_update(current_price, Variable(required_margin_rate))

        random_index = random.randint(0, len(agents) - 1)
        gen_loss = agents[random_index].effective_margin
        gen_gradient = generator.gradient(gen_loss)
        gen_gradients = gen_gradient

        generator.optimizer.update(generator.params, gen_gradients)

        disc_losses = []
        disc_gradients = []
        for agent in agents:
            # ここで明示的にVariableを使った損失構築（これにより.backward()で連鎖的に勾配計算可能に）
            disc_loss = mul(Variable(-1.0), agent.effective_margin)
            disc_losses.append(disc_loss)
            disc_gradient = agent.gradient(disc_loss)
            disc_gradients.append(disc_gradient)

            agent.optimizer.update(agent.params, disc_gradient)



        print(f"Generation {generation}, Gen Loss: {gen_loss.value}, Gradients: {gen_gradients}")

        history["disc_gradients"].append(disc_gradients)
        history["disc_losses"].append(disc_losses)
        history["generated_states"].append(generated_states)
        history["actions"].append(actions)
        history["agent_assets"].append([agent.effective_margin for agent in agents])
        history["liquidity"].append(current_liquidity)
        history["slippage"].append(current_slippage)
        history["gen_gradients"].append(gen_gradients)
        history["gen_loss"].append(gen_loss)
        history["scale_factor"].append(generator.scale_factor)

        if generation == generations // 2:
            use_rule_based = False

    # Calculate position value
    i = 0
    for agent in agents:
        position_value = 0
        if agent.positions and agent.margin_maintenance_flag==False:
            print("🔍 Before position_value calculation, positions:")
            print(agent.positions)
            # TensorArray を Python の list に変換
            #positions_tensor = agent.positions.stack()

            position_value += sum(size.value * (current_price.value - open_price.value) if status.value==1 else
                         -size.value * (current_price.value - open_price.value) if status.value==-1 else
                         0 for size, status, open_price, _, _, _ in agent.positions)
            #position_value = tf.reduce_sum(
            #    positions_tensor[:, 0] * tf.math.sign(positions_tensor[:, 1]) * (current_price - positions_tensor[:, 2])
            #    )

        else:
            position_value = 0

        print(f"{i}th agent")
        print(f"預託証拠金:{agent.margin_deposit.value}")
        print(f"有効証拠金:{agent.effective_margin.value}")
        print(f"ポジション損益:{position_value}")
        print(f"確定利益:{agent.realized_profit.value}")
        print(f"証拠金維持率:{agent.margin_maintenance_rate}")
        print(f"check total:{agent.margin_deposit.value+position_value}")
        print(f"ロスカットしたか:{agent.margin_maintenance_flag}")
        print("\n")
        i += 1

    # ファイルへの記録
    with open(f"./txt_dir/kabu_agent_based_metatraining_seed-{seed}_lsf-{initial_scale_factor.value}_generations-{generations}.json", "w") as f:
        json.dump(serialize(history), f, indent=2)