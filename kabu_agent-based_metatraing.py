import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class MarketGenerator:
    def __init__(self, input_dim=4, output_dim=3):
        self.model = tf.keras.Sequential([
            layers.Input(shape=(input_dim,)),  # 価格、流動性、スリッページ, 需給
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(output_dim, activation='linear')  # 価格、流動性、スリッページ
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

    def generate(self, inputs):
        """
        現在の市場状態を基に次の市場状態を生成
        """
        return self.model(inputs[np.newaxis, :])

    def train(self, discriminator_performance):
        """
        生成者を学習
        - `inputs`: 現在の市場状態
        - `discriminator_performance`: 各エージェントの総資産から計算される識別者の評価
        """
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(discriminator_performance)  # 識別が容易な場合に損失を増加
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


class RLAgent:
    def __init__(self, initial_cash=100000):
        self.cash_balance = initial_cash
        self.position = 0
        self.total_assets = initial_cash
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
        return self.model(state[np.newaxis, :]).numpy()[0][0]

    def update_assets(self, action, current_price):
        """
        資産とポジションを更新
        """
        if action > 0:  # 買い
            cost = action * current_price
            if cost <= self.cash_balance:
                self.cash_balance -= cost
                self.position += action
        elif action < 0:  # 売り
            sell_amount = min(abs(action), self.position)
            self.cash_balance += sell_amount * current_price
            self.position -= sell_amount
        self.total_assets = self.cash_balance + self.position * current_price

    def train(self):
        """
        エージェントを学習
        - `input_data`: 現在の市場状態（総資産、価格）
        """
        with tf.GradientTape() as tape:
            loss = -self.total_assets  # 総資産の最大化を目指す
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


# 初期化
num_agents = 5
agents = [RLAgent() for _ in range(num_agents)]
generator = MarketGenerator()
states = np.random.rand(3)  # 初期市場状態 [価格、流動性、スリッページ]
supply_and_demand = 0

# トレーニングループ
generations = 10
for generation in range(generations):
    actions = []
    input_data = np.append(states, supply_and_demand)  # 市場状態 + 需給
    generated_states = generator.generate(input_data).numpy()[0]  # 次の市場状態を生成
    states = generated_states[:3]  # 価格、流動性、スリッページ

    # 各エージェントの行動を決定
    for agent in agents:
        action = agent.act(np.array([agent.total_assets, states[0]]))  # 総資産、価格
        actions.append(action)
        agent.update_assets(action, states[0])  # 市場価格のみで資産更新

    # 需給を計算
    supply_and_demand = sum(actions)

    # エージェントの学習
    for agent in agents:
        agent.train(np.array([agent.total_assets, states[0]]))

    # 生成者の学習
    discriminator_performance = np.array([agent.total_assets for agent in agents])  # 総資産が高いほど識別困難
    generator.train(discriminator_performance)

    print(f"Generation {generation}, Best Agent Assets: {max(agent.total_assets for agent in agents):.2f}")
