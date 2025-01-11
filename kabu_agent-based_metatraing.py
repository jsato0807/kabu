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
        """
        生成者の学習
        """
        loss = tf.reduce_mean(discriminator_performance)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))



class RLAgent:
    def __init__(self, initial_cash=100000):
        self.cash_balance = tf.Variable([initial_cash], dtype=tf.float32)
        self.long_position = tf.Variable([0.0], dtype=tf.float32)
        self.short_position = tf.Variable([0.0], dtype=tf.float32)
        self.total_assets = tf.Variable([initial_cash], dtype=tf.float32)
        self.unfulfilled_buy_orders = tf.Variable([0.0], dtype=tf.float32)
        self.unfulfilled_sell_orders = tf.Variable([0.0], dtype=tf.float32)
        self.model = tf.keras.Sequential([
            layers.Dense(128, activation='sigmoid', input_dim=2),
            layers.Dense(64, activation='sigmoid'),
            layers.Dense(1, activation='linear')
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def act(self, state):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        action = self.model(state[tf.newaxis, :])[0][0]
        max_buy = self.cash_balance / state[1]
        max_sell = self.long_position
        return tf.clip_by_value(action, -tf.squeeze(max_sell), tf.squeeze(max_buy))

    def update_assets(self, action, current_price):
        """
        資産とポジションを更新
        """
        current_price = tf.convert_to_tensor(current_price, dtype=tf.float32)  # current_priceをテンソル化
        # 買い注文処理
        if action > 0:
            total_buy = action + self.unfulfilled_buy_orders
            cost = total_buy * current_price

            buy_condition = cost <= self.cash_balance
            fulfilled_buy = tf.where(buy_condition, total_buy, self.cash_balance / current_price)
            remaining_buy = tf.where(buy_condition, 0.0, total_buy - fulfilled_buy)
            self.cash_balance.assign(self.cash_balance - fulfilled_buy * current_price)
            self.long_position.assign(self.long_position + fulfilled_buy)
            self.unfulfilled_buy_orders.assign(remaining_buy)

        # 売り注文処理
        elif action < 0:
            total_sell = abs(action) + self.unfulfilled_sell_orders

            sell_condition = total_sell <= self.long_position
            fulfilled_sell = tf.where(sell_condition, total_sell, self.long_position)
            remaining_sell = tf.where(sell_condition, 0.0, total_sell - fulfilled_sell)

            self.cash_balance.assign(self.cash_balance + fulfilled_sell * current_price)
            self.long_position.assign(self.long_position - fulfilled_sell)
            self.unfulfilled_sell_orders.assign(remaining_sell)

            additional_short = remaining_sell - self.short_position
            self.short_position.assign(self.short_position + tf.where(~sell_condition, additional_short, -self.short_position))

        elif action == 0:
            self.cash_balance.assign(self.cash_balance + action * current_price)

        # 総資産を計算
        print(f"action:{action}")
        if action > 0:
            print(f"fulfilled_buy:{fulfilled_buy}")
        if action < 0:
            print(f"fulfilled_sell:{fulfilled_sell}")
        print(f"self.cash_balance:{self.cash_balance}")
        print(f"self.long_position:{self.long_position}")
        print(f"current_price:{current_price}")
        print(f"self.short_position:{self.short_position}")

        #exit()
        self.total_assets.assign(
            self.cash_balance + self.long_position * current_price - self.short_position * current_price
        )

    def train(self, tape):
        """
        エージェントの学習
        """
        loss = -self.total_assets  # 総資産の最大化
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


def distribute_unfilled_orders(supply_and_demand, agents):
    """
    未成立注文をエージェントにランダムに分配
    - `supply_and_demand`: 需給バランス（正なら買い注文が過剰、負なら売り注文が過剰）
    - `agents`: 全エージェントのリスト
    """
    num_agents = len(agents)
    if supply_and_demand > 0:
        # 売り注文は全て成立するが、買い注文の一部が未成立
        unfulfilled_orders = np.random.multinomial(
            supply_and_demand, [1 / num_agents] * num_agents
        )
        for agent, unfulfilled_buy in zip(agents, unfulfilled_orders):
            agent.unfulfilled_buy_orders.assign_add(tf.constant([float(unfulfilled_buy)], dtype=tf.float32))
    elif supply_and_demand < 0:
        # 買い注文は全て成立するが、売り注文の一部が未成立
        unfulfilled_orders = np.random.multinomial(
            abs(supply_and_demand), [1 / num_agents] * num_agents
        )
        for agent, unfulfilled_sell in zip(agents, unfulfilled_orders):
            agent.unfulfilled_sell_orders.assign_add(tf.constant([float(unfulfilled_sell)], dtype=tf.float32))




# 初期化
num_agents = 5
agents = [RLAgent() for _ in range(num_agents)]
generator = MarketGenerator()
# statesの初期化（適切な値に設定）
initial_price = 100.0  # 適切な初期価格
initial_liquidity = 1.0  # 流動性の初期値
initial_slippage = 0.01  # スリッページの初期値

states = tf.Variable([initial_price, initial_liquidity, initial_slippage], dtype=tf.float32)  # 初期市場状態 [価格、流動性、スリッページ]
supply_and_demand = tf.Variable([0.0], dtype=tf.float32)
volume = tf.Variable(0.0, dtype=tf.float32)
gamma = tf.constant(1.0, dtype=tf.float32)

use_rule_based = True  # 初期段階ではルールベースで流動性・スリッページを計算

# 記録用の辞書
history = {
    "generated_states": [],  # 生成者の出力: [価格, 流動性, スリッページ]
    "actions": [],     # 各エージェントの行動
    "agent_assets": [],       # 各エージェントの総資産
    "liquidity": [],
    "slippage" : [],
}

# トレーニングループ
generations = 10
for generation in range(generations):
    with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape:
        # 市場生成用の入力データ
        input_data = tf.concat([states, tf.reshape(supply_and_demand, [1])], axis=0)
        generated_states = generator.generate(input_data)[0]
        current_price, current_liquidity, current_slippage = tf.split(generated_states, num_or_size_splits=3)

        # 状態を更新
        states = tf.stack([current_price, current_liquidity, current_slippage])

        # ルールベースでの調整
        if use_rule_based:
            k = 1 / (1 + gamma * volume)
            current_liquidity = 1 / (1 + k * abs(supply_and_demand))
            current_slippage = abs(supply_and_demand) / (current_liquidity + 1e-6)

        # 各エージェントの行動
        actions = [agent.act([agent.total_assets, current_price]) for agent in agents]
        supply_and_demand = tf.reduce_sum(actions)

        # 未成立注文の分配
        distribute_unfilled_orders(supply_and_demand, agents)

        # 資産更新
        for agent, action in zip(agents, actions):
            agent.update_assets(action, current_price)

        # 識別者の評価（discriminator_performance）
        discriminator_performance = tf.stack([agent.total_assets for agent in agents])

        # 生成者の損失計算
        gen_loss = tf.reduce_mean(discriminator_performance)
        #print("Model trainable variables:", generator.model.trainable_variables)
        #print("Gen loss:", gen_loss)
        #exit()


        # 識別者の損失計算
        disc_losses = [-agent.total_assets for agent in agents]
        #for agent in agents:
        #    print("Model trainable variables:", agent.model.trainable_variables)
        #    print("disc losses:", disc_losses)
        #    exit()

        print(f"Generated States: {generated_states.numpy()}")
        print(f"Discriminator Performance: {discriminator_performance.numpy()}")
        print(f"Gen Loss: {gen_loss.numpy()}")

    # 勾配の計算
    # 生成者の勾配
    gen_gradients = gen_tape.gradient(gen_loss, generator.model.trainable_variables)
    print(f"Generator Gradients: {gen_gradients}")

    # 識別者の勾配
    for agent, disc_loss in zip(agents, disc_losses):
        disc_gradients = disc_tape.gradient(disc_loss, agent.model.trainable_variables)
        print(f"Agent Gradients: {disc_gradients}")

        # 識別者の重みを更新
        agent.optimizer.apply_gradients(zip(disc_gradients, agent.model.trainable_variables))

    # 生成者の重みを更新
    generator.optimizer.apply_gradients(zip(gen_gradients, generator.model.trainable_variables))

    # 各タイムステップのデータを記録
    history["generated_states"].append(generated_states.numpy())
    history["actions"].append(actions)
    history["agent_assets"].append([agent.total_assets.numpy() for agent in agents])
    history["liquidity"].append(current_liquidity.numpy())
    history["slippage"].append(current_slippage.numpy())

    print(f"Generation {generation}, Best Agent Assets: {max(float(agent.total_assets.numpy()) for agent in agents):.2f}")

    # 進化段階でルールベースを切り替え
    if generation == generations // 2:
        use_rule_based = False

    # ファイルへの記録
    with open("kabu_agent-based_metalearning.txt", "w") as f:
        f.write(str(history))

    os.chmod("kabu_agent-based_metalearning.txt", 0o444)
    print("ファイルを読み取り専用に設定しました")
