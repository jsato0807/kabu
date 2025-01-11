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


class RLAgent:
    def __init__(self, initial_cash=100000):
        self.cash_balance = tf.convert_to_tensor(initial_cash, dtype=tf.float32)
        self.long_position = tf.convert_to_tensor(0.0, dtype=tf.float32)
        self.short_position = tf.convert_to_tensor(0.0, dtype=tf.float32)
        self.total_assets = tf.convert_to_tensor(initial_cash, dtype=tf.float32)
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

    def update_assets(self, action, current_price):
        current_price = tf.convert_to_tensor(current_price, dtype=tf.float32)

        if action > 0:  # 買い注文処理
            total_buy = action + self.unfulfilled_buy_orders
            cost = total_buy * current_price

            buy_condition = tf.cast(cost <= self.cash_balance, dtype=tf.float32)
            fulfilled_buy = buy_condition * total_buy + (1 - buy_condition) * (self.cash_balance / (current_price + 1e-6))
            remaining_buy = (1 - buy_condition) * (total_buy - fulfilled_buy)

            self.cash_balance -= fulfilled_buy * current_price
            self.long_position += fulfilled_buy
            self.unfulfilled_buy_orders = remaining_buy

        elif action < 0:  # 売り注文処理
            total_sell = abs(action) + self.unfulfilled_sell_orders

            sell_condition = tf.cast(total_sell <= self.long_position, dtype=tf.float32)
            fulfilled_sell = sell_condition * total_sell + (1 - sell_condition) * self.long_position
            remaining_sell = (1 - sell_condition) * (total_sell - fulfilled_sell)

            self.cash_balance += fulfilled_sell * current_price
            self.long_position -= fulfilled_sell
            additional_short = remaining_sell - self.short_position
            self.short_position += (1 - sell_condition) * additional_short + sell_condition * -self.short_position
            self.unfulfilled_sell_orders = remaining_sell

        # 総資産を計算
        self.total_assets = self.cash_balance + self.long_position * current_price - self.short_position * current_price

    def train(self, tape):
        loss = -self.total_assets  # 総資産の最大化
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


def distribute_unfilled_orders(supply_and_demand, agents):
    num_agents = len(agents)
    if supply_and_demand > 0:
        unfulfilled_orders = np.random.multinomial(
            supply_and_demand, [1 / num_agents] * num_agents
        )
        for agent, unfulfilled_buy in zip(agents, unfulfilled_orders):
            agent.unfulfilled_buy_orders += float(unfulfilled_buy)
    elif supply_and_demand < 0:
        unfulfilled_orders = np.random.multinomial(
            abs(supply_and_demand), [1 / num_agents] * num_agents
        )
        for agent, unfulfilled_sell in zip(agents, unfulfilled_orders):
            agent.unfulfilled_sell_orders += float(unfulfilled_sell)


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
}

# トレーニングループ
generations = 10000
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
        actions = [agent.act([agent.total_assets, current_price]) for agent in agents]
        supply_and_demand = tf.reduce_sum(actions)

        volume = tf.reduce_sum([tf.abs(action) for action in actions])

        # 未成立注文の分配
        distribute_unfilled_orders(supply_and_demand, agents)

        # 資産更新
        for agent, action in zip(agents, actions):
            agent.update_assets(action, current_price)

        # 識別者の評価（discriminator_performance）
        discriminator_performance = tf.stack([agent.total_assets for agent in agents])

        # 生成者の損失計算
        gen_loss = tf.reduce_mean(discriminator_performance)
        disc_losses = [-agent.total_assets for agent in agents]

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
    history["agent_assets"].append([agent.total_assets.numpy() for agent in agents])
    history["liquidity"].append(current_liquidity.numpy())
    history["slippage"].append(current_slippage.numpy())

    print(f"Generation {generation}, Best Agent Assets: {max(float(agent.total_assets.numpy()) for agent in agents):.2f}")

    # 進化段階でルールベースを切り替え
    if generation == generations // 2:
        use_rule_based = False

# ファイルへの記録
with open("kabu_agent-based_metatraining.txt", "w") as f:
    f.write(str(history))
os.chmod("kabu_agent-based_metatraining.txt", 0o444)
print("ファイルを読み取り専用に設定しました")
