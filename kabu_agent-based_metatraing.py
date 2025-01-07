import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers
from deap import base, creator, tools, gp
from deap.tools import sortNondominated

# 市場の値動きを生成するパレートフロント対応遺伝的プログラミング
class MarketGeneratorPareto:
    def __init__(self, num_agents):
        # 適応度関数を進化させるための木構造設定
        self.pset = gp.PrimitiveSet("MAIN", 3)  # supply_and_demand, liquidity, slippage
        self.pset.addPrimitive(np.add, 2)
        self.pset.addPrimitive(np.subtract, 2)
        self.pset.addPrimitive(np.multiply, 2)
        self.pset.addEphemeralConstant("rand", lambda: random.uniform(-1, 1))

        # パレートフロント対応の多目的適応度を定義
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, 1.0, -1.0, 1.0, -1.0))  # 6基準
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=2)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr, pset=self.pset)
        self.toolbox.register("evaluate", self.evaluate_individual)

        self.population = self.toolbox.population(n=10)

        # 市場特性の初期化
        self.num_agents = num_agents
        self.market_characteristics = [
            ([random.uniform(-1, 1) for _ in range(num_agents)], random.uniform(0.1, 1), random.uniform(0.0, 0.5))
            for _ in range(100)
        ]

    def evaluate_individual(self, individual):
        """
        パレートフロント用の評価関数
        """
        func = gp.compile(expr=individual, pset=self.pset)

        supply_scores = []
        liquidity_scores = []
        slippage_scores = []

        for agent_actions, _, _, liquidity, _, slippage in self.market_characteristics:
            try:
                # エージェントの行動から需給を計算
                buy_orders = sum([max(0, action) for action in agent_actions])  # 買い注文量
                sell_orders = sum([abs(min(0, action)) for action in agent_actions])  # 売り注文量
                supply_and_demand = buy_orders - sell_orders

                func(supply_and_demand, liquidity, slippage)

                supply_scores.append(abs(supply_and_demand))  # 需給
                liquidity_scores.append(liquidity)  # 流動性
                slippage_scores.append(slippage)  # スリッページ
            except Exception:
                # エラー時にはペナルティを与える
                supply_scores.append(-100)
                liquidity_scores.append(-100)
                slippage_scores.append(-100)

        # 各基準の正負方向を評価
        return (
            np.mean(supply_scores),          # 正方向の需給
            -np.mean(supply_scores),         # 負方向の需給
            np.mean(liquidity_scores),       # 正方向の流動性
            -np.mean(liquidity_scores),      # 負方向の流動性
            np.mean(slippage_scores),        # 正方向のスリッページ
            -np.mean(slippage_scores),       # 負方向のスリッページ
        )

    def evolve(self,agent_actions):
        """
        パレートフロントに基づく進化
        """
        pareto_fronts = sortNondominated(self.population, len(self.population), first_front_only=False)
        next_generation = pareto_fronts[0]  # 最前列の個体群

        offspring = tools.selTournament(next_generation, len(self.population), tournsize=3)
        offspring = list(map(self.toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                self.toolbox.mate(child1, child2)
        for mutant in offspring:
            if random.random() < 0.2:
                self.toolbox.mutate(mutant)

        for ind in offspring:
            ind.fitness.values = self.toolbox.evaluate(ind)

        self.population[:] = offspring
        self.update_market_characteristics(agent_actions)

    def update_market_characteristics(self,agent_actions):
        """
        市場特性をエージェント行動に基づき動的に更新
        """
        new_characteristics = []

        for _, alpha, beta, liquidity, gamma, slippage in self.market_characteristics:
            # エージェントの行動から需給を計算
            buy_orders = sum([max(0, action) for action in agent_actions])
            sell_orders = sum([abs(min(0, action)) for action in agent_actions])
            supply_and_demand = buy_orders - sell_orders
            volume = buy_orders + sell_orders

            # 流動性の更新
            updated_liquidity = liquidity + alpha * volume - beta * supply_and_demand

            # スリッページの更新
            updated_slippage = gamma * slippage + (1 - gamma) * abs(supply_and_demand) / (updated_liquidity + 1e-6)


            new_characteristics.append((agent_actions, alpha, beta, liquidity, gamma ,updated_slippage))

        self.market_characteristics = new_characteristics

    def generate_market_dynamics(self):
        """
        パレートフロントに基づいて最適な市場モデルを生成
        """
        best_ind = tools.selBest(self.population, 1)[0]
        return gp.compile(best_ind, self.pset)




# 強化学習エージェント
class Agent:
    def __init__(self, input_dim):
        self.model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_dim=input_dim),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def train(self, market_dynamics, epochs=10):
        for _ in range(epochs):
            with tf.GradientTape() as tape:
                inputs = np.random.uniform(-1, 1, (100, 2))
                rewards = market_dynamics(inputs[:, 0], inputs[:, 1])
                predictions = self.model(inputs, training=True)
                loss = tf.reduce_mean(tf.square(predictions - rewards))
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def evaluate(self, market_dynamics):
        inputs = np.random.uniform(-1, 1, (100, 2))
        predictions = self.model(inputs, training=False)
        rewards = market_dynamics(inputs[:, 0], inputs[:, 1])
        return np.mean(rewards - predictions.numpy())


# メタ学習データの蓄積
class MetaLearner:
    def __init__(self):
        self.data = []

    def store_experience(self, market_dynamics, agent_performance):
        self.data.append((market_dynamics, agent_performance))

    def analyze(self):
        # 蓄積されたデータを分析（例: パフォーマンスの分布を確認）
        performances = [entry[1] for entry in self.data]
        print("Average Agent Performance:", np.mean(performances))


# メインプロセス
market_gen = MarketGeneratorPareto()
agents = [Agent(input_dim=2) for _ in range(5)]
meta_learner = MetaLearner()

for generation in range(10):
    market_dynamics = market_gen.generate_market_dynamics()
    agent_actions = []
    for agent in agents:
        agent.train(market_dynamics)
        agent_action = agent.evaluate(market_dynamics)
        agent_actions.append(agent_action)

    meta_learner.store_experience(market_dynamics, agent_actions)

    print(f"Generation {generation}, Best Market Dynamics Evaluated")
    market_gen.evolve(agent_actions)

meta_learner.analyze()
