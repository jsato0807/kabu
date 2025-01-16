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

    def train(self, tape, discriminator_performance):
        loss = tf.reduce_mean(discriminator_performance)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


def check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min):
    if effective_margin_max < effective_margin:
        effective_margin_max = effective_margin
    if effective_margin_min > effective_margin:
        effective_margin_min = effective_margin
    return effective_margin_max, effective_margin_min


def update_margin_maintenance_rate(effective_margin, required_margin, margin_cut_threshold):
    if required_margin != 0:  
        margin_maintenance_rate = (effective_margin / required_margin) * 100
    else:
        margin_maintenance_rate = np.inf

    if margin_maintenance_rate <= margin_cut_threshold:
        print(f"Margin maintenance rate is {margin_maintenance_rate}%, below threshold. Forced liquidation triggered.")
        return True, margin_maintenance_rate  # フラグと値を返す
    return False, margin_maintenance_rate  # フラグと値を返す

class RLAgent:
    def __init__(self, initial_cash=100000):
        self.positions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        self.closed_positions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        self.positions_index = tf.convert_to_tensor(0.0, dtype=tf.float32)
        self.closed_positions_index = tf.convert_to_tensor(0.0, dtype=tf.float32)
        self.effective_margin = tf.convert_to_tensor(initial_cash, dtype=tf.float32)
        self.required_margin = tf.convert_to_tensor(0.0, dtype=tf.float32)
        self.margin_deposit = tf.convert_to_tensor(initial_cash, dtype=tf.float32)
        self.long_position = tf.convert_to_tensor(0.0, dtype=tf.float32)
        self.short_position = tf.convert_to_tensor(0.0, dtype=tf.float32)
        self.unfulfilled_buy_orders = tf.convert_to_tensor(0.0, dtype=tf.float32)
        self.unfulfilled_sell_orders = tf.convert_to_tensor(0.0, dtype=tf.float32)
        self.model = tf.keras.Sequential([
            layers.Dense(128, activation='sigmoid', input_dim=2),
            layers.Dense(64, activation='sigmoid'),
            layers.Dense(1, activation='linear')
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def act(self, state):
        # state の形状を統一して結合
        state = [tf.reshape(tf.convert_to_tensor(s, dtype=tf.float32), [1]) for s in state]
        state = tf.concat(state, axis=0)
        state = tf.reshape(state, [1, -1])  # 形状を統一
        action = self.model(state)[0][0]
        return action
    
    def _remove_position(self, index):
        """
        Remove a position by replacing it with an empty placeholder.
        """
        empty_pos = tf.constant([], dtype=tf.float32)  # 空のテンソル
        self.positions = self.positions.write(index, empty_pos)


    def update_assets(self, long_order_size, short_order_size, long_close_position, short_close_position, current_price, total_buy_demand, total_sell_supply, required_margin_rate=0.04, margin_cut_threshold=100.0):
        """
        資産とポジションの更新を実行
        """
        current_price = tf.convert_to_tensor(current_price, dtype=tf.float32)

        # --- 新規注文処理 ---
        def process_new_order(order_size, trade_type, margin_rate):
            if order_size > 0:
                order_margin = order_size * current_price * margin_rate
                margin_maintenance_flag, margin_maintenance_rate = update_margin_maintenance_rate(self.effective_margin, self.required_margin, margin_cut_threshold)
                if margin_maintenance_flag:
                    print(f"Margin cut triggered during {'Buy' if trade_type == 1.0 else 'Sell'} order processing.")
                    return
                order_capacity = self.effective_margin - (self.required_margin + order_margin)
                if order_capacity < 0:
                    print(f"Cannot process {'Buy' if trade_type == 1.0 else 'Sell'} order due to insufficient order capacity.")
                    return
                if margin_maintenance_rate > 100 and order_capacity > 0:
                    add_required_margin = current_price * order_size * margin_rate
                    self.required_margin += add_required_margin
                    pos = tf.stack([self.positions_index, order_size, trade_type, current_price, 0.0, add_required_margin])
                    self.positions = self.positions.write(self.positions_index, pos)
                    self.positions_index += 1
                    print(f"Opened {'Buy' if trade_type == 1.0 else 'Sell'} position at {current_price}, Effective Margin: {self.effective_margin}")

        # Process new buy and sell orders
        process_new_order(long_order_size, 1.0, required_margin_rate)  # Buy
        process_new_order(short_order_size, -1.0, required_margin_rate)  # Sell

        # --- ポジション決済処理 ---
        pos_id_max = self.positions_index - 1  # 現在の最大 ID
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

            # 決済ロジック
            fulfilled_size = tf.minimum(close_position, size)
            self.effective_margin += profit - unrealized_profit
            self.margin_deposit += profit
            self.realized_profit += profit
            self.required_margin -= margin * (fulfilled_size / size)

            # 部分決済または完全決済の処理
            size -= fulfilled_size
            if size > 0:  # 部分決済の場合
                pos = tf.stack([pos_id, size, pos_type, open_price, unrealized_profit, margin * (size / fulfilled_size)])
                self.positions = self.positions.write(i, pos)
                pos = tf.stack([pos_id, fulfilled_size, pos_type, open_price, 0, 0])
                self.closed_positions = self.closed_positions.write(self.closed_positions_index, pos)
                self.closed_positions_index += 1
            else:  # 完全決済の場合
                pos = tf.stack([pos_id, fulfilled_size, pos_type, open_price, 0, 0])
                self.closed_positions = self.closed_positions.write(self.closed_positions_index, pos)
                self.closed_positions_index += 1
                self._remove_position(i)

            if pos_type == 1.0:
                long_close_position -= fulfilled_size
            elif pos_type == -1.0:
                short_close_position -= fulfilled_size

        # --- 含み益の更新 ---
        pos_id_max = self.positions_index - 1  # 現在の最大 ID
        for pos_id in range(pos_id_max + 1):  # 最大 ID までの範囲を網羅
            pos = self.positions.read(pos_id)
            if tf.size(pos) == 0:  # 空ポジションはスキップ
                continue
            pos_id, size, pos_type, open_price, _, margin = tf.unstack(pos)

            unrealized_profit = size * (current_price - open_price) if pos_type == 1.0 else size * (open_price - current_price)
            self.effective_margin += unrealized_profit - pos[4]
            add_required_margin = -pos[5] + current_price * size * required_margin_rate
            self.required_margin += add_required_margin

            pos = tf.stack([pos_id, size, pos_type, open_price, unrealized_profit, margin+add_required_margin])
            self.positions = self.positions.write(pos_id, pos)

        # --- 強制ロスカットのチェック ---
        margin_maintenance_flag, _ = update_margin_maintenance_rate(self.effective_margin, self.required_margin, margin_cut_threshold)
        if margin_maintenance_flag:
            print("Forced margin cut triggered.")
            pos_id_max = self.positions_index - 1  # 現在の最大 ID
            for pos_id in range(pos_id_max + 1):  # 最大 ID までの範囲を網羅
                pos = self.positions.read(pos_id)
                self.closed_positions = self.closed_positions.write(self.closed_positions_index, pos)
                self.closed_positions_index += 1
            self.positions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            self.required_margin = 0.0



    def train(self, tape):
        loss = -self.effective_margin  # 総資産の最大化
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))



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
generations = 1000
use_rule_based = True  # 初期段階ではルールベースで流動性・スリッページを計算
gamma = tf.convert_to_tensor(1,dtype=tf.float32)
volume = tf.convert_to_tensor(0,dtype=tf.float32)

for generation in range(generations):
    with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape:
        # 市場生成用の入力データ
        input_data = tf.concat([tf.reshape(states, [-1]), [supply_and_demand]], axis=0)
        generated_states = generator.generate(input_data)[0]
        current_price, current_liquidity, current_slippage = tf.split(generated_states, num_or_size_splits=3)

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
        actions = [agent.act([agent.effective_margin, current_price]) for agent in agents]
        print(f"actions:{actions}")
        supply_and_demand = tf.reduce_sum(actions)

        volume = tf.reduce_sum([tf.abs(action) for action in actions])


        # 資産更新
        for agent, action in zip(agents, actions):
            agent.update_assets(action, current_price)

        # 識別者の評価（discriminator_performance）
        discriminator_performance = tf.stack([agent.effective_margin for agent in agents])

        # 生成者の損失計算
        gen_loss = tf.reduce_mean(discriminator_performance)
        disc_losses = [-agent.effective_margin for agent in agents]

    # 勾配の計算
    # 生成者の勾配
    gen_gradients = gen_tape.gradient(gen_loss, generator.model.trainable_variables)
    generator.optimizer.apply_gradients(zip(gen_gradients, generator.model.trainable_variables))

    # 識別者の勾配
    for agent, disc_loss in zip(agents, disc_losses):
        disc_gradients = disc_tape.gradient(disc_loss, agent.model.trainable_variables)
        agent.optimizer.apply_gradients(zip(disc_gradients, agent.model.trainable_variables))

    # 記録用の辞書に状態を追加
    history["generated_states"].append(generated_states.numpy())
    history["actions"].append(actions)
    history["agent_assets"].append([agent.effective_margin.numpy() for agent in agents])
    history["liquidity"].append(current_liquidity.numpy())
    history["slippage"].append(current_slippage.numpy())
    history["gen_gradients"].append(gen_gradients)
    for disc_loss in disc_losses:
        history["disc_gradients"].append(disc_tape.gradient(disc_loss, agent.model.trainable_variables))

    #print(f"Generation {generation}, Best Agent Assets: {max(float(agent.effective_margin.numpy()) for agent in agents):.2f}")
    #print(f"gen_gradients:{gen_gradients}")
    #exit()
    for disc_loss in disc_losses:
        print(f"disc_gradients:{disc_tape.gradient(disc_loss, agent.model.trainable_variables)}")

    exit()

    # 進化段階でルールベースを切り替え
    if generation == generations // 2:
        use_rule_based = False

# ファイルへの記録
with open("./txt_dir/kabu_agent-based_metatraining.txt", "w") as f:
    f.write(str(history))
os.chmod("./txt_dir/kabu_agent-based_metatraining.txt", 0o444)
print("ファイルを読み取り専用に設定しました")
