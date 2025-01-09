import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class MarketGenerator:
    def __init__(self, input_dim=5, output_dim=3):
        """
        input_dim: 現在の市場状態（価格、流動性、スリッページ、需給）
        output_dim: 次の市場状態（価格、流動性、スリッページ）
        """
        self.model = tf.keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(output_dim, activation='linear')  # 出力: 価格、流動性、スリッページ
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
        self.long_position = tf.Variable(0.0, dtype=tf.float32)  # 買いポジション
        self.short_position = tf.Variable(0.0, dtype=tf.float32)  # 売りポジション
        self.total_assets = tf.Variable(initial_cash, dtype=tf.float32)
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
        action = self.model(state[tf.newaxis, :])[0][0]  # TensorFlow形式の計算を保持
        # キャッシュやポジションを超えないように制約を追加
        max_buy = self.cash_balance / state[1]  # 現在価格で買える最大量
        max_sell = self.long_position  # 売却可能な最大量（ロングポジションのみ）
        return tf.clip_by_value(action, -max_sell, max_buy)

    def update_assets(self, action, current_price):
        """
        エージェントの資産とポジションを更新
        """
        if action > 0:  # 買い
            cost = action * current_price
            buy_condition = cost <= self.cash_balance  # 購入可能条件
            self.cash_balance.assign_sub(tf.where(buy_condition, cost, 0.0))
            self.long_position.assign_add(tf.where(buy_condition, action, 0.0))
        elif action < 0:  # 売り
            sell_amount = abs(action)
            sell_condition = sell_amount <= self.long_position  # ロングポジション売却条件
            self.cash_balance.assign_add(tf.where(sell_condition, sell_amount * current_price, 0.0))
            self.long_position.assign_sub(tf.where(sell_condition, sell_amount, 0.0))

            # ショートポジションの処理
            additional_short = sell_amount - self.long_position
            self.short_position.assign_add(tf.where(~sell_condition, additional_short, 0.0))

        # 資産総額を計算
        self.total_assets.assign(
            self.cash_balance
            + self.long_position * current_price
            - self.short_position * current_price
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
    未成立の注文をランダムにエージェントに配分
    """
    num_agents = len(agents)
    if supply_and_demand > 0:  # 売りが不足して買い注文が未成立
        unfulfilled_orders = np.random.multinomial(
            supply_and_demand, [1 / num_agents] * num_agents
        )
        for agent, unfulfilled in zip(agents, unfulfilled_orders):
            agent.unfilled_orders.assign(agent.unfilled_orders + unfulfilled)
    elif supply_and_demand < 0:  # 買いが不足して売り注文が未成立
        unfulfilled_orders = np.random.multinomial(
            abs(supply_and_demand), [1 / num_agents] * num_agents
        )
        for agent, unfulfilled in zip(agents, unfulfilled_orders):
            agent.unfilled_orders.assign(agent.unfilled_orders - unfulfilled)



# 初期化
num_agents = 5
agents = [RLAgent() for _ in range(num_agents)]
generator = MarketGenerator()
states = tf.Variable(np.random.rand(3), dtype=tf.float32)  # 初期市場状態 [価格、流動性、スリッページ]
supply_and_demand = tf.Variable(0.0, dtype=tf.float32)
volume = tf.Variable(0.0, dtype=tf.float32)
gamma = tf.constant(1.0, dtype=tf.float32)

use_rule_based = True  # 初期段階ではルールベースで流動性・スリッページを計算

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
        k = 1/(1+gamma*volume)
        current_liquidity = 1 / (1 + k*abs(supply_and_demand))
        current_slippage = abs(supply_and_demand) / (current_liquidity + 1e-6)

    # 各エージェントの行動を決定
    for agent in agents:
        action = agent.act(tf.stack([agent.total_assets, current_price]))
        actions.append(action)

    # 需給を計算
    supply_and_demand = tf.reduce_sum(actions)
    distribute_unfilled_orders(supply_and_demand, agents)  # 未成立注文をランダムに配分

    #取引量を計算
    volume = tf.reduce_sum(abs(action) for action in actions)

    for agent in agents:
        agent.update_assets(action, current_price,supply_and_demand)

    # エージェントの学習
    for agent in agents:
        agent.train()

    # 生成者の学習
    discriminator_performance = tf.stack([agent.total_assets for agent in agents])
    generator.train(discriminator_performance)

    # 状態を更新
    states = tf.stack([current_price, current_liquidity, current_slippage])

    print(f"Generation {generation}, Best Agent Assets: {max(agent.total_assets for agent in agents):.2f}")

    # 進化段階でルールベースを切り替え
    if generation == generations // 2:
        use_rule_based = False
