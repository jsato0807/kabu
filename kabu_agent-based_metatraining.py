import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os

class MarketGenerator:
    def __init__(self, input_dim=4, output_dim=3):
        self.model = tf.keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation='sigmoid'),
            layers.Dense(64, activation='sigmoid'),
            layers.Dense(output_dim, activation='relu')
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

    def generate(self, inputs):
        return self.model(tf.expand_dims(inputs, axis=0))

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

class RLAgent:
    def __init__(self, initial_cash=100000):
        self.positions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True,clear_after_read=False)
        self.closed_positions = []
        self.positions_index = tf.Variable(0, dtype=tf.float32)
        self.effective_margin = tf.Variable(initial_cash, dtype=tf.float32)
        self.required_margin = 0
        self.margin_deposit = tf.Variable(initial_cash, dtype=tf.float32)
        self.realized_profit = tf.Variable(0.0, dtype=tf.float32)
        self.long_position = tf.Variable(0.0, dtype=tf.float32)
        self.short_position = tf.Variable(0.0, dtype=tf.float32)
        self.unfulfilled_buy_orders = tf.Variable(0.0, dtype=tf.float32)
        self.unfulfilled_sell_orders = tf.Variable(0.0, dtype=tf.float32)
        self.model = tf.keras.Sequential([
            layers.Dense(128, activation='sigmoid', input_dim=2),
            layers.Dense(64, activation='sigmoid'),
            layers.Dense(4, activation='relu')
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def act(self, state):
        # state の形状を統一して結合
        state = [tf.reshape(tf.Variable(s, dtype=tf.float32), [1]) for s in state]
        state = tf.concat(state, axis=0)
        state = tf.reshape(state, [1, -1])  # 形状を統一
        action = self.model(state)
        return action
    
    def _remove_position(self, index):
        """
        Remove a position by replacing it with an empty placeholder.
        """
        empty_pos = tf.zeros((6,), dtype=tf.float32)  # 現在のポジションの形状に一致させる
        self.positions = self.positions.write(tf.cast(index, tf.int32), empty_pos)


    def process_new_order(self, order_size, trade_type, current_price, margin_rate):
        if order_size > 0:
            if trade_type == 1:
                order_margin = (order_size + self.unfulfilled_buy_orders * current_price * margin_rate)
            if trade_type == -1:
                order_margin = (order_size + self.unfulfilled_sell_orders * current_price * margin_rate)
            margin_maintenance_flag, margin_maintenance_rate = update_margin_maintenance_rate(self.effective_margin, self.required_margin)
            if margin_maintenance_flag:
                print(f"Margin cut triggered during {'Buy' if trade_type == 1.0 else 'Sell'} order processing.")
                return
            order_capacity = (self.effective_margin - (self.required_margin+ order_margin)).numpy()
            if order_capacity < 0:
                print(f"Cannot process {'Buy' if trade_type == 1.0 else 'Sell'} order due to insufficient order capacity.")
                return
            if margin_maintenance_rate > 100 and order_capacity > 0:
                add_required_margin = current_price * order_size * margin_rate
                self.required_margin += add_required_margin.numpy()
                pos = tf.stack([self.positions_index, order_size, trade_type, current_price, 0.0, add_required_margin])
                self.positions = self.positions.write(tf.cast(self.positions_index, tf.int32), pos)

                self.positions_index.assign(self.positions_index + 1)
                print(f"Opened {'Buy' if trade_type==1 else ('Sell' if trade_type == -1 else 'Unknown')} position at {current_price}, required margin: {self.required_margin}")
                margin_maintenance_flag, margin_maintenance_rate = update_margin_maintenance_rate(self.effective_margin,self.required_margin)
                if margin_maintenance_flag:
                    print(f"margin maintenance rate is {margin_maintenance_rate},so loss cut is executed in process_new_order, effective_margin: {self.effective_margin}")
                    return


    def update_assets(self, long_order_size, short_order_size, long_close_position, short_close_position, current_price, required_margin_rate=0.04):
        """
        資産とポジションの更新を実行
        """

        # --- 新規注文処理 ---
        # Process new buy and sell orders
        self.process_new_order(long_order_size, 1.0, current_price, required_margin_rate)  # Buy
        self.process_new_order(short_order_size, -1.0, current_price, required_margin_rate)  # Sell

        # --- ポジション決済処理 ---
        pos_id_max = int(self.positions_index - 1)  # 現在の最大 ID
        for pos_id in range(pos_id_max + 1):  # 最大 ID までの範囲を網羅
            pos = self.positions.read(pos_id)
            if tf.size(pos) == 0:  # 空ポジションはスキップ
                continue
            pos_id, size, pos_type, open_price, unrealized_profit, margin = tf.unstack(pos)

            if pos_type == 1.0 and long_close_position > 0:  # Buy
                close_position = long_close_position
                profit = close_position * (current_price - open_price)
            elif pos_type == -1.0 and short_close_position > 0:  # Sell
                close_position = short_close_position
                profit = close_position * (open_price - current_price)
            else:
                continue
            print(f"close_position:{close_position}, current_price:{current_price},open_price:{open_price}")

            # 決済ロジック
            fulfilled_size = tf.minimum(close_position, size)
            self.effective_margin.assign(self.effective_margin + profit - unrealized_profit)
            self.margin_deposit.assign(self.margin_deposit + profit)
            self.realized_profit.assign(self.realized_profit + profit)
            add_required_margin = -margin * (fulfilled_size / size)
            self.required_margin += add_required_margin.numpy()

            # 部分決済または完全決済の処理
            size -= fulfilled_size
            if size > 0:  # 部分決済の場合
                pos = tf.stack([pos_id, size, pos_type, open_price, unrealized_profit, margin * (1 - fulfilled_size / size)])
                self.positions = self.positions.write(tf.cast(pos_id, tf.int32), pos)
                pos = tf.stack([pos_id, fulfilled_size, pos_type, open_price, 0, 0])
                self.closed_positions.append(pos)

            else:  # 完全決済の場合
                pos = tf.stack([pos_id, fulfilled_size, pos_type, open_price, 0, 0])
                self.closed_positions.append(pos)

                self._remove_position(pos_id)
            print(f"Closed {'Buy' if pos_type==1 else ('Sell' if pos_type == -1 else 'Unknown')} position at {current_price} with profit {profit} ,grid {open_price}, Effective Margin: {self.effective_margin}, Required Margin: {self.required_margin}")

            if pos_type == 1.0:
                self.unfulfilled_buy_orders.assign(long_close_position - fulfilled_size)
            elif pos_type == -1.0:
                self.unfulfilled_sell_orders.assign(short_close_position - fulfilled_size)

            margin_maintenance_flag, margin_maintenance_rate = update_margin_maintenance_rate(self.effective_margin,self.required_margin)
            if margin_maintenance_flag:
                print(f"margin maintenance rate is {margin_maintenance_rate},so loss cut is executed in position closure process, effective_margin: {self.effective_margin}")
                return

        # --- 含み益の更新 ---
        pos_id_max = int(self.positions_index - 1)  # 現在の最大 ID
        for pos_id in range(pos_id_max + 1):  # 最大 ID までの範囲を網羅
            pos = self.positions.read(pos_id)
            if tf.size(pos) == 0:  # 空ポジションはスキップ
                continue
            pos_id, size, pos_type, open_price, before_unrealized_profit, margin = tf.unstack(pos)

            unrealized_profit = size * (current_price - open_price) if pos_type == 1.0 else size * (open_price - current_price)
            self.effective_margin.assign(self.effective_margin + unrealized_profit - before_unrealized_profit)
            add_required_margin = -margin + current_price * size * required_margin_rate
            self.required_margin += add_required_margin.numpy()

            pos = tf.stack([pos_id, size, pos_type, open_price, unrealized_profit, margin+add_required_margin])
            self.positions = self.positions.write(tf.cast(pos_id, tf.int32), pos)
            print(f"unrealized_profit:{unrealized_profit}, before_unrealized_profit:{before_unrealized_profit}")
            print(f"updated effective margin against price {current_price} , Effective Margin: {self.effective_margin}")

            margin_maintenance_flag, margin_maintenance_rate = update_margin_maintenance_rate(self.effective_margin,self.required_margin)
            if margin_maintenance_flag:
                print(f"margin maintenance rate is {margin_maintenance_rate},so loss cut is executed in position  process, effective_margin: {self.effective_margin}")
                return

        # --- 強制ロスカットのチェック ---
        margin_maintenance_flag, _ = update_margin_maintenance_rate(self.effective_margin, self.required_margin)
        if margin_maintenance_flag:
            print("Forced margin cut triggered.")
            pos_id_max = self.positions_index - 1  # 現在の最大 ID
            for pos_id in range(pos_id_max + 1):  # 最大 ID までの範囲を網羅
                pos = self.positions.read(pos_id)
                self.closed_positions.append(pos)

                self._remove_position(pos_id)
            #self.required_margin = 0.0

            # 全ポジションをクリア
            self.positions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            self.positions_index.assign(0)

            self.required_margin = 0


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



# トレーニングループ
num_agents = 5
agents = [RLAgent() for _ in range(num_agents)]
generator = MarketGenerator()

states = tf.constant([100.0, 1.0, 0.01], dtype=tf.float32)
supply_and_demand = tf.constant(0.0, dtype=tf.float32)

# 記録用の辞書
history = {
    "generated_states": [],  # 生成者の出力: [価格, 流動性, スリッページ]
    "actions": [],     # 各エージェントの行動
    "agent_assets": [],       # 各エージェントの総資産
    "liquidity": [],
    "slippage" : [],
    "gen_gradients": [],
    "disc_gradients": [],
}

# トレーニングループ
generations = 10
use_rule_based = True  # 初期段階ではルールベースで流動性・スリッページを計算
gamma = tf.Variable(1,dtype=tf.float32)
volume = tf.Variable(0,dtype=tf.float32)
actions = tf.TensorArray(dtype=tf.float32, size=len(agents),dynamic_size=True)
disc_losses = tf.TensorArray(dtype=tf.float32, size=len(agents), dynamic_size=True)
initial_losses = tf.TensorArray(dtype=tf.float32, size=len(agents), dynamic_size=True)


for generation in range(generations):
    #if generation == 1:
    #    print(f"generation is 1")
    #    exit()
    with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape:
        # 市場生成用の入力データ
        input_data = tf.concat([tf.reshape(states, [-1]), [supply_and_demand]], axis=0)
        # 各要素に 1e-6 を加算して対数を取る
        log_inputs = tf.math.log(input_data + 1e-6)
        generated_states = generator.generate(log_inputs)[0]
        unlog_generated_states = tf.math.exp(generated_states) - 1e-6
        current_price, current_liquidity, current_slippage = tf.split(unlog_generated_states, num_or_size_splits=3)

        # 状態を更新
        current_price = tf.reshape(current_price, [])
        current_liquidity = tf.reshape(current_liquidity, [])
        current_slippage = tf.reshape(current_slippage, [])

        # ルールベースでの調整
        if use_rule_based:
            k = 1 / (1 + gamma * volume)
            current_liquidity = 1 / (1 + k * abs(supply_and_demand))
            current_slippage = abs(supply_and_demand) / (current_liquidity + 1e-6)

        # states の更新
        states = tf.stack([current_price, current_liquidity, current_slippage])

        # 各エージェントの行動
        #actions = [agent.act([agent.effective_margin, current_price]) for agent in agents]
        i = 0
        for agent in agents:
            # 各要素に 1e-6 を加算して対数を取る
            log_inputs = tf.math.log([x + 1e-6 for x in [agent.effective_margin, current_price]])
            unlog_action = tf.math.exp(agent.act(log_inputs)) - 1e-6
            actions = actions.write(i,unlog_action)
            i += 1
        print(f"actions:{actions.stack()}")

        supply_and_demand = tf.reduce_sum(actions.stack())

        volume = tf.reduce_sum([tf.abs(actions.stack())])

        # 資産更新
        print(actions.stack().shape)
        for agent, action in zip(agents, actions.stack()):
            # アクションの形状 (1, 4) を (4,) に変換
            action_flat = tf.reshape(action, [-1])  # 形状 (4,)
            # 各項目を変数に分解
            long_order_size, short_order_size, long_close_position, short_close_position = tf.unstack(action_flat)
            agent.update_assets(long_order_size, short_order_size, long_close_position, short_close_position, current_price)

        # 識別者の評価（discriminator_performance）
        discriminator_performance = tf.stack([agent.effective_margin for agent in agents])

        # 生成者の損失計算
        if generation < generations//2:
            i = 0
            for action in (actions.stack()):
                action_flat = tf.reshape(action,[-1])
                long_order_size, short_order_size, long_close_position, short_close_position = tf.unstack(action_flat)

                initial_loss = (tf.math.log(current_price + 1e-6) \
                            + tf.math.log(long_order_size + 1e-6) \
                            + tf.math.log(short_order_size + 1e-6) \
                            + tf.math.log(long_close_position + 1e-6) \
                            + tf.math.log(short_close_position + 1e-6))
                initial_losses = initial_losses.write(i,initial_loss)
                disc_losses = disc_losses.write(i,-initial_loss)
                i += 1

            

            gen_loss = tf.reduce_mean(initial_losses.stack())
            print(disc_losses.stack().shape)
            #exit()
            stacked_disc_losses = disc_losses.stack()

        elif generation >= generations // 2:
        #if generation >= 0:
            gen_loss = tf.reduce_mean(discriminator_performance)
            i = 0
            for agent in agents:
                disc_losses = disc_losses.write(i,-agent.effective_margin)
                i += 1
            stacked_disc_losses = disc_losses.stack()

        # 勾配の計算
        # 生成者の勾配
        print(type(gen_loss))
        gen_gradients = gen_tape.gradient(gen_loss, generator.model.trainable_variables)
        #print(type(gen_gradients))
        print(f"gen_loss: {gen_loss}")
        print(f"gen_gradients: {gen_gradients}")

        generator.optimizer.apply_gradients(zip(gen_gradients, generator.model.trainable_variables))

        # 識別者の勾配
        print(f"disc_losses: {stacked_disc_losses}, type: {type(disc_losses)}") 
        disc_gradients = []
        i = 0
        for agent, disc_loss in zip(agents, stacked_disc_losses):
            print(i)
            print(f"disc_loss: {disc_loss}, type: {type(disc_loss)}") 
            print(f"disc_losses: {stacked_disc_losses}, type: {type(stacked_disc_losses)}")  
            #print(type(disc_loss))
            disc_gradient = disc_tape.gradient(disc_loss, agent.model.trainable_variables)
            #print(type(disc_gradient))
            disc_gradients.append(disc_gradient)
            print(f"disc_gradient:{disc_gradient}")
            #exit()
            agent.optimizer.apply_gradients(zip(disc_gradient, agent.model.trainable_variables))
            i += 1

        print(f"gen_gradients: {gen_gradients}")

        # 記録用の辞書に状態を追加
        history["disc_gradients"].append(disc_gradients.numpy())

    history["generated_states"].append(generated_states.numpy())
    history["actions"].append(actions)
    history["agent_assets"].append([agent.effective_margin.numpy() for agent in agents])
    history["liquidity"].append(current_liquidity.numpy())
    history["slippage"].append(current_slippage.numpy())
    history["gen_gradients"].append(gen_gradients)

    #print(f"Generation {generation}, Best Agent Assets: {max(float(agent.effective_margin.numpy()) for agent in agents):.2f}")
    #print(f"gen_gradients:{gen_gradients}")
    #exit()

    #exit()

    # 進化段階でルールベースを切り替え
    if generation == generations // 2:
        use_rule_based = False

    print(f"generation:{generation}")
    print(" ")
    print(" ")
    print(" ")
    print(" ")

# ファイルへの記録
with open("./txt_dir/kabu_agent-based_metatraining.txt", "w") as f:
    f.write(str(history))
os.chmod("./txt_dir/kabu_agent-based_metatraining.txt", 0o444)
print("ファイルを読み取り専用に設定しました")
