import numpy as np
import math
import sign
import os
import random
from collections import OrderedDict
from common.layers2 import *


    

seed = 0
initial_log_scale_factor = np.log(np.exp(1))

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)  # Python の乱数シード
    np.random.seed(seed)  # NumPy の乱数シード

class MarketGenerator:
    def __init__(self, input_size, hidden_size, output_size):
        # ランダム初期化（正規分布）
        self.params = OrderedDict()
        self.params['W1'] = Variable(random.gauss(0, 0.01 * math.sqrt(1/input_size)))
        self.params['b1'] = Variable(0.0)
        self.params['W2'] = Variable(random.gauss(0, 0.01 * math.sqrt(1/hidden_size)))
        self.params['b2'] = Variable(0.0)
        self.params['W3'] = Variable(random.gauss(0, 0.01 * math.sqrt(1/hidden_size)))
        self.params['b3'] = Variable(0.0)
        self.params['log_scale_factor'] = Variable(log(exp(1)))

        self.layers = OrderedDict()
        self.layers['Affine1'] = lambda x: affine(x, self.params['W1'], self.params['b1'])
        self.layers['Sigmoid1'] = sigmoid
        self.layers['Affine2'] = lambda x: affine(x, self.params['W2'], self.params['b2'])
        self.layers['Sigmoid2'] = sigmoid
        self.layers['Affine3'] = lambda x: affine(x, self.params['W3'], self.params['b3'])
        self.layers['Softplus'] = softplus
        self.layers['LogScale'] = lambda x: mul(x, self.params['log_scale_factor'].value)

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
            original_value = param.value

            # f(x + h)
            param.value = original_value + h
            fxh1 = loss_func(self.predict(x)).value

            # f(x - h)
            param.value = original_value - h
            fxh2 = loss_func(self.predict(x)).value

            # numerical gradient
            grad = (fxh1 - fxh2) / (2 * h)
            grads[name] = grad

            # reset param
            param.value = original_value

        return grads



    """
    def train(self, tape, discriminator_performance,generation,actions):
        long_order_size, short_order_size, long_close_position, short_close_position = tf.unstack(actions)
        loss = tf.math.log(current_price + 1e-6) \
     + tf.math.log(long_order_size + 1e-6) \
     + tf.math.log(short_order_size + 1e-6) \
     + tf.math.log(long_close_position + 1e-6) \
     + tf.math.log(short_close_position + 1e-6)
        if generation == generations // 2:
            loss = tf.reduce_mean(discriminator_performance)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    """
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
        self.params['W1'] = Variable(random.gauss(0, math.sqrt(1 / input_size)))
        self.params['b1'] = Variable(0.0)
        self.params['W2'] = Variable(random.gauss(0, math.sqrt(1 / hidden_size)))
        self.params['b2'] = Variable(0.0)
        self.params['W3'] = Variable(random.gauss(0, math.sqrt(1 / hidden_size)))
        self.params['b3'] = Variable(0.0)

        self.layers = OrderedDict()
        self.layers['Affine1'] = lambda x: affine(x, self.params['W1'], self.params['b1'])
        self.layers['Sigmoid1'] = sigmoid
        self.layers['Affine2'] = lambda x: affine(x, self.params['W2'], self.params['b2'])
        self.layers['Sigmoid2'] = sigmoid
        self.layers['Affine3'] = lambda x: affine(x, self.params['W2'], self.params['b2'])
        self.layers['ReLU3'] = relu

    def predict(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x

    def gradient(self, loss):
        loss.backward()
        grads = {
            name: param.grad(param)
            for name, param in self.params.items()
        }
        return grads

    def numerical_gradient(self, x, reward_func):
        def loss_fn():
            y = self.predict(x)
            return reward_func(y)

        h = 1e-4
        grads = {}
        for name, param in self.params.items():
            orig = param.value

            # f(x + h)
            param.value = orig + h
            loss_plus = loss_fn().value

            # f(x - h)
            param.value = orig - h
            loss_minus = loss_fn().value

            # numerical gradient
            grad = (loss_plus - loss_minus) / (2 * h)
            grads[name] = grad

            # restore original value
            param.value = orig

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
                self.positions.remove(pos)

            #    self.unfulfilled_short_open = short_close_position - fulfilled_size

            self.margin_maintenance_flag, self.margin_maintenance_rate = update_margin_maintenance_rate(self.effective_margin,self.required_margin)
            if self.margin_maintenance_flag:
                print(f"margin maintenance rate is {self.margin_maintenance_rate},so loss cut is executed in position closure process, effective_margin: {self.effective_margin}")
                continue
        #for pos_id in sorted(to_be_removed, reverse=True):  # 降順で削除（pos_id がズレないように）
        #    self._remove_position(pos_id)


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

            if pos_type == 1:
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
            for pos_id in range(len(self.positions)):
                try:
                    pos = self.positions[pos_id]
                    size, pos_type, open_price, before_unrealized_profit, margin, _ = pos
                except:
                    continue

                if pos_type == 1:
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
                self.positions.remove(pos)


            self.required_margin = 0
            _, self.margin_maintenance_rate = update_margin_maintenance_rate(
                self.effective_margin, self.required_margin
            )


        #"""


    """
    def train(self, tape,generation,actions):
        long_order_size, short_order_size, long_close_position, short_close_position = tf.unstack(actions)
        loss = -(tf.math.log(current_price + 1e-6) \
     + tf.math.log(long_order_size + 1e-6) \
     + tf.math.log(short_order_size + 1e-6) \
     + tf.math.log(long_close_position + 1e-6) \
     + tf.math.log(short_close_position + 1e-6))
        if generation == generations // 2:
            loss = -self.effective_margin  # 総資産の最大化
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    """



def match_orders(agents, actions, current_price, required_margin_rate):
    # 1️⃣ 各エージェントの注文を取得
    long_open_orders = []
    short_open_orders = []
    long_close_orders = []
    short_close_orders = []

    for agent, action in zip(agents, actions):
        action_flat = np.reshape(action, [-1])
        long_open, short_open, long_close, short_close = action_flat
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


if __name__ == "__main__":
    # トレーニングループ
    num_agents = 5
    set_seed(seed)
    agents = [RLAgent() for _ in range(num_agents)]
    generator = MarketGenerator()

    states = [100.0, 1.0, 0.01]
    supply_and_demand = 0.0

    # 記録用の辞書
    history = {
        "generated_states": [],  # 生成者の出力: [価格, 流動性, スリッページ]
        "actions": [],     # 各エージェントの行動
        "agent_assets": [],       # 各エージェントの総資産
        "liquidity": [],
        "slippage" : [],
        "gen_gradients": [],
        "disc_gradients": [],
        "gen_loss": [],
        "disc_losses": [],
        "log_scale_factor": [],
    }

    # トレーニングループ
    generations = 165
    use_rule_based = True  # 初期段階ではルールベースで流動性・スリッページを計算
    required_margin_rate=0.04
    gamma = 1
    volume = 0
    actions = []
    disc_losses = []
    gen_losses = []


    for generation in range(generations):
        set_seed(generation)

        ## 各要素に 1e-6 を加算して対数を取る
        # 供給需要の符号付き log 変換
        log_supply_and_demand = sign(supply_and_demand) * math.log(abs(supply_and_demand) + 1e-6)

        # generator の入力データ
        log_inputs = np.concat([
            math.log(np.reshape(states,[-1]) + 1e-6),
            [log_supply_and_demand]
        ], axis=0)

        generated_states = generator.generate(log_inputs)[0]
        unlog_generated_states = math.exp(generated_states) - 1e-6
        current_price, current_liquidity, current_slippage = np.split(unlog_generated_states, num_or_size_splits=3)


        # ルールベースでの調整
        if use_rule_based:
            # k を 計算
            k = 1/(1+gamma*volume)

            # current_liquidity を 計算
            current_liquidity = 1/(1+k*abs(supply_and_demand))

            # current_slippage を 計算
            current_slippage = abs(supply_and_demand)/(current_liquidity + 1e-6)


        # states の更新
        states = current_price, current_liquidity, current_slippage]

        # 各エージェントの行動
        #actions = [agent.act([agent.effective_margin, current_price]) for agent in agents]
        i = 0
        for agent in agents:
            # 各要素に 1e-6 を加算して対数を取る
            log_inputs = math.log([agent.effective_margin + 1e-6, current_price + 1e-6])
            # ネットワークの出力を処理し、1e-6 を減算
            unlog_action = math.exp(agent.predict(log_inputs)) - 1e-6
            actions.append(unlog_action)
            i += 1
        #print(f"actions:{actions.stack()}")

        volume = sum_variables(abs(a) for a in actions)

        # 資産更新
        #print(actions.stack().shape)
        i = 0
        match_orders(agents, actions, current_price, required_margin_rate)

        supply_and_demand = sum_variables([
            agent.unfulfilled_long_open
            - agent.unfulfilled_short_open
            - agent.unfulfilled_long_close
            + agent.unfulfilled_short_close
            for agent in agents
        ])
        print(f"supply_and_demand:{supply_and_demand}")

        for agent, action in zip(agents, actions):
            # アクションの形状 (1, 4) を (4,) に変換
            action_flat = np.reshape(action, [-1])  # 形状 (4,)
            # 各項目を変数に分解
            long_order_size, short_order_size, long_close_position, short_close_position = action_flat

            #reset calculation graph and update effective_margin by using outputs of models with multiplying 0 in order not to change the value of effective_margin itself. 

            print("\n")
            print(f"update positions of {i}th agent")
            print(f"long_order_size:{long_order_size}, short_order_size:{short_order_size}")
            #agent.update_assets(long_order_size, short_order_size, long_close_position, short_close_position, current_price)
            #agent.process_new_order(long_order_size,short_order_size,current_price,required_margin_rate)
            #agent.process_position_closure(long_close_position,short_close_position,current_price)
            agent.process_position_update(current_price,required_margin_rate)
            #agent.update_effective_margin = current_price * (long_order_size + short_order_size + long_close_position + short_close_position)
            print(f"{i}th long_order_size:{long_order_size}")
            print(f"{i}th short_order_size:{short_order_size}")
            print(f"{i}th long_close_position:{long_close_position}")
            print(f"{i}th short_close_position:{short_close_position}")
            i += 1
            #print(f"current_price:{current_price}")
            #print(f"update_effective_margin:{agent.update_effective_margin}")

        # 識別者の評価（discriminator_performance）
        #disc_tape.watch([agent.effective_margin for agent in agents])  # 必要に応じて追跡
        #discriminator_performance = tf.stack([agent.effective_margin for agent in agents])

        # 生成者の損失計算
        #if generation < generations//2:
        #    i = 0
        #    for action in (actions.stack()):
        #        action_flat = tf.reshape(action,[-1])
        #        long_order_size, short_order_size, long_close_position, short_close_position = tf.unstack(action_flat)
    #
        #        initial_loss = (tf.math.log(current_price + 1e-6) \
        #                    + tf.math.log(long_order_size + 1e-6) \
        #                    + tf.math.log(short_order_size + 1e-6) \
        #                    + tf.math.log(long_close_position + 1e-6) \
        #                    + tf.math.log(short_close_position + 1e-6))
        #        initial_losses = initial_losses.write(i,initial_loss)
        #        disc_losses = disc_losses.write(i,-initial_loss)
        #        i += 1
    #
    #            
    #
    #         gen_loss = tf.reduce_mean(initial_losses.stack())
    #         print(disc_losses.stack().shape)
    #         #exit()
    #         stacked_disc_losses = disc_losses.stack()

        #elif generation >= generations // 2:
        if generation >= 0:
            #gen_loss = tf.reduce_mean(discriminator_performance)
            # エージェントの effective_margin を TensorFlow のテンソルとして格納
            effective_margins = [agent.effective_margin for agent in agents]
            # ランダムなインデックスを取得
            random_index = random.randint(0, len(agents) - 1)
            # ランダムに選択した effective_margin を取得
            selected_margin = effective_margins[random_index]
            print(f"selected agent is {random_index}th agent")
            #gen_loss = tf.reduce_mean(tf.stack([agent.effective_margin for agent in agents]))
            gen_loss = selected_margin
            #print(f"gen_loss:{gen_loss}")
            i = 0
            for agent in agents:
                disc_losses.append(-agent.effective_margin)
                i += 1
            #stacked_disc_losses = disc_losses.stack()


            """
            #this code is useful for check whether a variable holds the information of calculation graph
            i = 0
            for agent in agents:
                pos_id_max = int(agent.positions_index - 1)  # 現在の最大 ID
                for pos_id in range(pos_id_max + 1):  # 最大 ID までの範囲を網羅
                    try:
                        pos = agent.positions.read(pos_id)
                        if tf.reduce_all(pos==0.0):
                            continue
                    except:
                        continue
                    size, pos_type, open_price, before_unrealized_profit, margin, _ = tf.unstack(pos)
                    print(f"margin:{margin}")
                    if pos_id >= 1:
                        break

                #disc_losses = disc_losses.write(i,size)
                gen_losses = gen_losses.write(i,generator.log_scale_factor)
            i += 1

        print(f"disc_losses:{disc_losses.stack().numpy()}")
        print(f"gen_losses:{gen_losses.stack().numpy()}")

        gen_loss = tf.reduce_mean(gen_losses.stack())
        """

        # 勾配の計算
        # 生成者の勾配
        gen_gradients = gen_tape.gradient(gen_loss, [generator.log_scale_factor] + generator.model.trainable_variables)

        gen_gradients = generator.gradient()

        print(f"gen_gradients:{gen_gradients}")
        print(f"gen_loss: {gen_loss}")
        #print(f"generation:{generation}")
        #print(f"gen_gradients: {gen_gradients}")
        #exit()

        generator.optimizer.apply_gradients(zip(gen_gradients, [generator.log_scale_factor] + generator.model.trainable_variables))

        # 識別者の勾配
        #print(f"disc_losses: {stacked_disc_losses}, type: {type(disc_losses)}") 
        disc_gradients = []
        i = 0
        #for agent, disc_loss in zip(agents, stacked_disc_losses):
        disc_losses_list = []
        for agent in agents:
            disc_loss = disc_losses.read(i)
            disc_losses_list.append(disc_loss)
            #print(f"disc_loss:{disc_loss}")
            #exit()
            #print(i)
            #print(f"disc_loss: {disc_loss}, type: {type(disc_loss)}") 
            #print(f"disc_losses: {stacked_disc_losses}, type: {type(stacked_disc_losses)}")  
            #print(type(disc_loss))
            disc_gradient = disc_tape.gradient(disc_loss, agent.model.trainable_variables)
            print(f"{i}th agents' disc_gradient:{disc_gradient}")

            #print(type(disc_gradient))
            disc_gradients.append(disc_gradient)
            #print(f"disc_gradient:{disc_gradient}")
            #exit()
            agent.optimizer.apply_gradients(zip(disc_gradient, agent.model.trainable_variables))
            i += 1

        #print(f"gen_gradients: {gen_gradients}")

        #print("gen_tape variables:", gen_tape.watched_variables())
        #print("disc_tape variables:", disc_tape.watched_variables())
        #exit()

        #for agent in agents:
        #    agent.effective_margin = agent.update_effective_margin

        print(f"trainable_variables:{generator.trainable_variables}")
        print(f"generation:{generation}")
        print(f"log_scale_factor:{generator.log_scale_factor.numpy()}")
        print(f"current_price:{current_price}")
        print(" ")
        print(" ")
        print(" ")
        print(" ")

        # 記録用の辞書に状態を追加
        history["disc_gradients"].append(disc_gradients)
        history["disc_losses"].append(disc_losses_list)
        history["generated_states"].append(generated_states.numpy())
        history["actions"].append(stacked_actions)
        history["agent_assets"].append([agent.effective_margin.numpy() for agent in agents])
        history["liquidity"].append(current_liquidity.numpy())
        history["slippage"].append(current_slippage.numpy())
        history["gen_gradients"].append(gen_gradients)
        history["gen_loss"].append(gen_loss)
        history["log_scale_factor"].append(generator.log_scale_factor.numpy())

    #print(f"Generation {generation}, Best Agent Assets: {max(float(agent.effective_margin.numpy()) for agent in agents):.2f}")
    #print(f"gen_gradients:{gen_gradients}")
    #exit()

    #exit()

    ## 進化段階でルールベースを切り替え
    #if generation == generations // 2:
    #    use_rule_based = False


    # Calculate position value
    i = 0
    for agent in agents:
        position_value = 0
        if agent.positions and agent.margin_maintenance_flag==False:
            print("🔍 Before position_value calculation, positions:")
            print(agent.positions.stack())
            # TensorArray を Python の list に変換
            #positions_tensor = agent.positions.stack()
            positions_list = agent.positions.stack().numpy().tolist()

            position_value += sum_variables(size * (current_price - open_price) if status==1 else
                         -size * (current_price - open_price) if status==-1 else
                         0 for size, status, open_price, _, _, _ in positions_list)
            #position_value = tf.reduce_sum(
            #    positions_tensor[:, 0] * tf.math.sign(positions_tensor[:, 1]) * (current_price - positions_tensor[:, 2])
            #    )

        else:
            position_value = 0

        print(f"{i}th agent")
        print(f"預託証拠金:{agent.margin_deposit.numpy()}")
        print(f"有効証拠金:{agent.effective_margin}")
        print(f"ポジション損益:{position_value}")
        print(f"確定利益:{agent.realized_profit.numpy()}")
        print(f"証拠金維持率:{agent.margin_maintenance_rate}")
        print(f"check total:{agent.margin_deposit+position_value}")
        print(f"ロスカットしたか:{agent.margin_maintenance_flag}")
        print("\n")
        i += 1

    # ファイルへの記録
    with open(f"./txt_dir/kabu_agent_based_metatraining_seed-{seed}_lsf-{log_scale_factor}_generations-{generations}.txt", "w") as f:
        f.write(str(history))