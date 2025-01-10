import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os

class MarketGenerator:
    def __init__(self, input_dim=4, output_dim=3):
        """
        input_dim: 現在の市場状態（価格、流動性、スリッページ、需給）
        output_dim: 次の市場状態（価格、流動性、スリッページ）
        """
        self.model = tf.keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(output_dim, activation='relu')  # 出力: 価格、流動性、スリッページ
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

    def generate(self, inputs):
        """
        現在の市場状態を基に次の市場状態を生成
        """
        return self.model(tf.expand_dims(inputs, axis=0))

    def train(self, discriminator_performance):
        """
        生成者を学習
        """
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(discriminator_performance)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


class RLAgent:
    def __init__(self, initial_cash=100000):
        self.cash_balance = tf.Variable(initial_cash, dtype=tf.float32)
        self.long_position = tf.Variable(0.0, dtype=tf.float32)
        self.short_position = tf.Variable(0.0, dtype=tf.float32)  # 売り注文の未成立分
        self.total_assets = tf.Variable(initial_cash, dtype=tf.float32)
        self.unfulfilled_buy_orders = tf.Variable(0.0, dtype=tf.float32)  # 未決済の買い注文
        self.unfulfilled_sell_orders = tf.Variable(0.0, dtype=tf.float32)  # 未決済の売り注文
        self.model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_dim=2),  # 総資産、価格
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='linear')  # 行動（買い/売り）
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def act(self, state):
        """
        現在の状態を基に行動を決定
        """
        action = self.model(state[tf.newaxis, :])[0][0]
        max_buy = self.cash_balance / state[1]  # 現在価格で買える最大量
        max_sell = self.long_position  # 売却可能な最大量
        return tf.clip_by_value(action, -max_sell, max_buy)

    def update_assets(self, action, current_price):
        """
        資産とポジションを更新
        """
        # 買い注文処理
        if action > 0:
            total_buy = action + self.unfulfilled_buy_orders  # 未成立分を追加
            cost = total_buy * current_price

            buy_condition = cost <= self.cash_balance
            fulfilled_buy = tf.where(buy_condition, total_buy, self.cash_balance / current_price)
            remaining_buy = tf.where(buy_condition, 0.0, total_buy - fulfilled_buy)

            self.cash_balance.assign_sub(fulfilled_buy * current_price)
            self.long_position.assign_add(fulfilled_buy)
            self.unfulfilled_buy_orders.assign(remaining_buy)

        # 売り注文処理
        elif action < 0:
            total_sell = abs(action) + self.unfulfilled_sell_orders  # 未成立分を追加

            sell_condition = total_sell <= self.long_position
            fulfilled_sell = tf.where(sell_condition, total_sell, self.long_position)
            remaining_sell = tf.where(sell_condition, 0.0, total_sell - fulfilled_sell)

            self.cash_balance.assign_add(fulfilled_sell * current_price)
            self.long_position.assign_sub(fulfilled_sell)
            self.unfulfilled_sell_orders.assign(remaining_sell)

            # 未成立売り注文をショートポジションとして記録
            additional_short = remaining_sell - self.short_position
            self.short_position.assign_add(tf.where(~sell_condition, additional_short, -self.short_position))

        # 総資産を更新
        self.total_assets.assign(
            self.cash_balance + self.long_position * current_price - self.short_position * current_price
        )






    def train(self):
        """
        エージェントを学習
        """
        with tf.GradientTape() as tape:
            loss = -self.total_assets  # 総資産の最大化を目指す
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
            agent.unfulfilled_buy_orders.assign_add(float(unfulfilled_buy))
    elif supply_and_demand < 0:
        # 買い注文は全て成立するが、売り注文の一部が未成立
        unfulfilled_orders = np.random.multinomial(
            abs(supply_and_demand), [1 / num_agents] * num_agents
        )
        for agent, unfulfilled_sell in zip(agents, unfulfilled_orders):
            agent.unfulfilled_sell_orders.assign_add(float(unfulfilled_sell))




# 初期化
num_agents = 5
agents = [RLAgent() for _ in range(num_agents)]
generator = MarketGenerator()
states = tf.Variable(np.random.rand(3), dtype=tf.float32)  # 初期市場状態 [価格、流動性、スリッページ]
supply_and_demand = tf.Variable(0.0, dtype=tf.float32)
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
    actions = []
    input_data = tf.concat([states, tf.expand_dims(supply_and_demand, axis=0)], axis=0)

    # 市場生成
    generated_states = generator.generate(input_data)[0]
    current_price, current_liquidity, current_slippage = tf.split(generated_states, num_or_size_splits=3)

    if use_rule_based:
        # ルールベースで流動性・スリッページを計算
        k = 1 / (1 + gamma * volume)
        current_liquidity = 1 / (1 + k * abs(supply_and_demand))
        current_slippage = abs(supply_and_demand) / (current_liquidity + 1e-6)

    # 各エージェントの行動を決定
    for agent in agents:
        action = agent.act(tf.stack([agent.total_assets, current_price]))
        actions.append(action)

    # 需給を計算
    supply_and_demand = tf.reduce_sum(actions)
    distribute_unfilled_orders(supply_and_demand, agents)  # 未成立注文をランダムに配分

    # 取引量を計算
    volume = tf.reduce_sum([tf.abs(action) for action in actions])

    # 各エージェントの資産を更新
    for agent, action in zip(agents, actions):
        agent.update_assets(action, current_price)

    # エージェントの学習
    for agent in agents:
        agent.train()

    # 生成者の学習
    discriminator_performance = tf.stack([agent.total_assets for agent in agents])
    generator.train(discriminator_performance)

    # 状態を更新
    states = tf.stack([current_price, current_liquidity, current_slippage])

    # 各タイムステップのデータを記録
    history["generated_states"].append(generated_states.numpy())
    history["actions"].append(actions)
    history["agent_assets"].append([agent.total_assets.numpy() for agent in agents])
    history["liquidity"].append(current_liquidity.numpy())
    history["slippage"].append(current_slippage.numpy())

    print(f"Generation {generation}, Best Agent Assets: {max(agent.total_assets.numpy() for agent in agents):.2f}")

    # 進化段階でルールベースを切り替え
    if generation == generations // 2:
        use_rule_based = False


    with open("kabu_agent-based_metalearning.txt","w") as f:
        f.write(str(history))

    os.chmod("kabu_agent-based_metalearning.txt",0o444)

    print("ファイルを読み取り専用に設定しました")