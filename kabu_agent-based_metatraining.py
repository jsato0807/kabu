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

    def update_assets(self, long_order_size,short_order_size, long_close_position, short_close_position, current_price, total_buy_demand, total_sell_supply,required_margin_rate=0.04, margin_cut_threshold=100.0):
        """
        資産とポジションの更新を実行
        """
        current_price = tf.convert_to_tensor(current_price, dtype=tf.float32)
    
        # Buy/Sell Fill Rates の計算
        buy_fill_rate = tf.minimum(1.0, total_sell_supply / (total_buy_demand + 1e-6))  # 買い注文の成立率
        sell_fill_rate = tf.minimum(1.0, total_buy_demand / (total_sell_supply + 1e-6))  # 売り注文の成立率 


        if  long_order_size > 0:
            order_margin = long_order_size * current_price * required_margin_rate
            margin_maintenance_flag, margin_maintenance_rate = update_margin_maintenance_rate(self.effective_margin,self.required_margin,margin_cut_threshold)
            if margin_maintenance_flag:
                return
            order_capacity = self.effective_margin - (self.required_margin + order_margin)
            if order_capacity < 0:
                order_capacity_flag = True
                print(f'cannot order because of lack of order capacity')
                return
            if margin_maintenance_rate > 100 and order_capacity > 0:
                #self.margin_deposit -= order_size * grid
                #self.effective_margin -= order_size * grid
                order_margin -= long_order_size * current_price * required_margin_rate
                add_required_margin = current_price * long_order_size * required_margin_rate
                self.required_margin += add_required_margin
                self.positions.append([long_order_size, i, 'Buy', current_price, 0, add_required_margin,date,0,0])
                print(f"Opened Buy position at {current_price}, Effective Margin: {self.effective_margin}")
                #break
                margin_maintenance_flag, margin_maintenance_rate = update_margin_maintenance_rate(self.effective_margin,self.required_margin,margin_cut_threshold)                                   
                if margin_maintenance_flag:
                        print("executed loss cut")
                        return

        if  short_order_size > 0:
            order_margin = short_order_size * current_price * required_margin_rate
            margin_maintenance_flag, margin_maintenance_rate = update_margin_maintenance_rate(self.effective_margin,self.required_margin, margin_cut_threshold)
            if margin_maintenance_flag:
                return
            order_capacity = self.effective_margin - (self.required_margin + order_margin)
            if order_capacity < 0:
                order_capacity_flag = True
                print(f'cannot order because of lack of order capacity')
                return
            if margin_maintenance_rate > 100 and order_capacity > 0:
                #self.margin_deposit -= order_size * grid
                #self.effective_margin -= order_size * grid
                order_margin -= short_order_size * current_price * required_margin_rate
                add_required_margin = current_price * short_order_size * required_margin_rate
                self.required_margin += add_required_margin
                self.positions.append([short_order_size, i, 'Buy', current_price, 0, add_required_margin,date,0,0])
                print(f"Opened Buy position at {current_price}, Effective Margin: {self.effective_margin}")
                #break
                margin_maintenance_flag, margin_maintenance_rate = update_margin_maintenance_rate(self.effective_margin,self.required_margin,margin_cut_threshold)                                   
                if margin_maintenance_flag:
                        print("executed loss cut")
                        return

                    

        # Position closure processing
        for pos in self.positions[:]:   #this means FIFO because you deal with positions in order 
                if long_close_position > 0:
                    long_fulfilled_size = min(long_close_position, pos[0])
                    profit = long_fulfilled_size * (current_price - pos[3])
                    self.effective_margin += profit - pos[4]
                    effective_margin_max, effective_margin_min = check_min_max_effective_margin(self.effective_margin, effective_margin_max, effective_margin_min)
                    self.margin_deposit += profit
                    self.realized_profit += profit
                    pos[7] += profit
                    self.required_margin -= pos[5] * long_fulfilled_size/pos[0]
                    pos[5] -= pos[5] * long_fulfilled_size/pos[0]
                    # 決済済みポジションを履歴に保存
                    closed_pos = pos.copy()  # オブジェクト参照を避けるためコピー
                    closed_pos[0] = long_fulfilled_size  # 決済済みのサイズに更新
                    closed_pos[2] = "Buy-Closed"
                    self.closed_positions.append(closed_pos)
            
                    # ポジションの更新
                    if long_fulfilled_size == pos[0]:  # ポジションが完全に決済された場合
                        self.positions.remove(pos)
                    else:  # 部分決済の場合
                        pos[0] -= long_fulfilled_size  # 残りのサイズを更新
            
                    long_close_position -= long_fulfilled_size  # 未決済量を更新
                    print(f"Closed Sell position at {current_price} with profit {profit} ,open_price {pos[3]}, Effective Margin: {self.effective_margin}")
                    #break

                    margin_maintenance_flag, margin_maintenance_rate = update_margin_maintenance_rate(self.effective_margin,self.required_margin,margin_cut_threshold)
                    if margin_maintenance_flag:
                            print("executed loss cut")
                            continue

                if short_close_position > 0:
                    short_fulfilled_size = min(short_close_position, pos[0])
                    profit = short_fulfilled_size * (current_price - pos[3])
                    self.effective_margin += profit - pos[4]
                    effective_margin_max, effective_margin_min = check_min_max_effective_margin(self.effective_margin, effective_margin_max, effective_margin_min)
                    self.margin_deposit += profit

                    self.realized_profit += profit
                    pos[7] += profit
                    self.required_margin -= pos[5] * long_fulfilled_size/pos[0]
                    pos[5] -= pos[5] * long_fulfilled_size/pos[0]
                    # 決済済みポジションを履歴に保存
                    closed_pos = pos.copy()  # オブジェクト参照を避けるためコピー
                    closed_pos[0] = short_fulfilled_size  # 決済済みのサイズに更新
                    closed_pos[2] = "Sell-Closed"
                    self.closed_positions.append(closed_pos)
            
                    # ポジションの更新
                    if short_fulfilled_size == pos[0]:  # ポジションが完全に決済された場合
                        self.positions.remove(pos)
                    else:  # 部分決済の場合
                        pos[0] -= short_fulfilled_size  # 残りのサイズを更新
            
                    short_close_position -= short_fulfilled_size  # 未決済量を更新
                    print(f"Closed Sell position at {current_price} with profit {profit} ,grid {pos[3]}, Effective Margin: {self.effective_margin}")
                    #break

                    margin_maintenance_flag, margin_maintenance_rate = update_margin_maintenance_rate(self.effective_margin,self.required_margin,margin_cut_threshold)
                    if margin_maintenance_flag:
                            print("executed loss cut")
                            continue


        #含み益の更新
        for pos in self.positions[:]:
            if pos[2] == 'Buy':
                unrealized_profit = pos[0] * (current_price - pos[3])
            elif pos[2] == "Sell":
                unrealized_profit = pos[0] * (pos[3] - current_price)
            
            self.effective_margin += unrealized_profit -  pos[4]  # Adjust for previous unrealized profit
            effective_margin_max, effective_margin_min = check_min_max_effective_margin(self.effective_margin, effective_margin_max, effective_margin_min)
            add_required_margin = -pos[5] + current_price * pos[0] * required_margin_rate
            self.required_margin += add_required_margin
            pos[4] = unrealized_profit  # Store current unrealized profit in the position
            pos[5] += add_required_margin
            print(f"updated effective margin against price {current_price} , Effective Margin: {self.effective_margin}")

            margin_maintenance_flag, margin_maintenance_rate = update_margin_maintenance_rate(self.effective_margin,self.required_margin,margin_cut_threshold)
            if margin_maintenance_flag:
                    print("executed loss cut")
                    continue
    

        # 強制ロスカットのチェック
        if margin_maintenance_flag:
            for pos in self.positions:
                if pos[2] == 'Sell' or pos[2] == 'Buy':
                    if pos[2] == 'Sell':
                        profit = - (current_price - pos[3]) * short_order_size  # 現在の損失計算
                    if pos[2] == 'Buy':
                        profit = (current_price - pos[3]) * long_order_size
                    self.effective_margin += profit - pos[4] # 損失分を証拠金に反映
                    effective_margin_max, effective_margin_min = check_min_max_effective_margin(self.effective_margin, effective_margin_max, effective_margin_min)
                    self.margin_deposit += profit
                    self.realized_profit += profit
                    pos[7] += profit
                    self.required_margin -= pos[5]
                    pos[5] = 0
                    pos[2] += "-Forced-Closed"
                    self.positions.remove(pos)
                    self.closed_positions.append(pos)
                    print(f"Forced Closed at {current_price} with grid {pos[3]}, Effective Margin: {self.effective_margin}")
                    _, margin_maintenance_rate = update_margin_maintenance_rate(self.effective_margin,self.required_margin,margin_cut_threshold)   


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
