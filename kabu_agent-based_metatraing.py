import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class MarketGenerator:
    def __init__(self, input_dim=4, output_dim=3):
        self.model = tf.keras.Sequential([
            layers.Input(shape=(input_dim,)),#価格、流動性、スリッページ, 需給
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(output_dim, activation='linear')  # 価格、流動性、スリッページ
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

    def generate(self, inputs):
        return self.model(inputs[np.newaxis, :])

    def train(self, discriminator_loss):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(discriminator_loss)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


class RLAgent:
    def __init__(self, initial_cash=100000):
        self.cash_balance = initial_cash
        self.position = 0
        self.total_assets = initial_cash
        self.model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_dim=2), #2 means total_assets, current_price
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='linear')  # 行動（買い/売り）
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def act(self, state):
        return self.model(state[np.newaxis, :])

    def update_assets(self, action, current_price):
        if action > 0:
            cost = action * current_price
            if cost <= self.cash_balance:
                self.cash_balance -= cost
                self.position += action
        elif action < 0:
            sell_amount = min(abs(action), self.position)
            self.cash_balance += sell_amount * current_price
            self.position -= sell_amount
        self.total_assets = self.cash_balance + self.position * current_price

    def train(self, input_data):
        with tf.GradientTape() as tape:
            predicted_action = self.model(input_data, training=True)
            self.update_assets(predicted_action,input_data[1])
            loss = - self.total_assets
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


# 初期化
num_agents = 5
agents = [RLAgent() for _ in range(num_agents)]
generator = MarketGenerator()
states = np.random.rand(3)
supply_and_demand = 0

# トレーニングループ
generations = 10
for generation in range(generations):
    actions = []
    input_data = np.append(states, supply_and_demand)
    states = generator.generate(input_data).numpy()[0]
    input_data = states
    for agent in agents:
        action = agent.act([agent.total_assets,states[0]])
        actions.append(action)
        agent.update_assets(action,states[0])  # 市場価格のみ

    supply_and_demand = sum(actions)
    #volume = sum(abs(x) for x in actions)

    # 識別者（エージェント）の更新
    for agent in agents:
        agent.train(np.array([agent.total_assets, states[0]]))

    # 生成者の更新
    generator_loss = np.mean([agent.total_assets for agent in agents])
    generator.train(generator_loss)

    print(f"Generation {generation}, Best Agent Assets: {max(agent.total_assets for agent in agents):.2f}")
