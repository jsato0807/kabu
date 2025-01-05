import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
from deap import base, creator, tools, gp

# 市場の値動きを生成する遺伝的プログラミング
class MarketGenerator:
    def __init__(self):
        # 適応度関数を進化させるための木構造設定
        self.pset = gp.PrimitiveSet("MAIN", 4)  # supply_and_demand, α, β, slippage
        self.pset.addPrimitive(np.add, 2)
        self.pset.addPrimitive(np.subtract, 2)
        self.pset.addPrimitive(np.multiply, 2)
        self.pset.addPrimitive(np.negative, 1)
        self.pset.addEphemeralConstant("rand", lambda: random.uniform(-1, 1))

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=2)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr, pset=self.pset)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.evaluate_fitness_function)

        self.population = self.toolbox.population(n=10)

        # 初期の市場特性を生成
        self.market_characteristics = [
            (random.uniform(0, 1, num_agent), random.uniform(0.1, 1), random.uniform(0, 0.5)) for _ in range(100)
        ]

    def evaluate_fitness_function(self, individual):
        """
        適応度関数自体を評価
        """
        func = gp.compile(individual, self.pset)

        # 適応度関数の評価（エージェントのパフォーマンスを基準）
        scores = []
        for agent_actions, alpha, beta, slippage in self.market_characteristics:
            # エージェントの行動から需給バランスと取引量を計算
            buy_orders = sum([max(0, action) for action in agent_actions])  # 買い注文量
            sell_orders = sum([abs(min(0, action)) for action in agent_actions])  # 売り注文量

            # 需給バランス D(t)
            supply_and_demand = buy_orders - sell_orders

            score = func(supply_and_demand, alpha, beta, slippage)
            if not np.isnan(score):  # 不正な計算結果を除外
                scores.append(score)

        return np.mean(scores),

    def evolve(self):
        """
        GPによる進化
        """
        offspring = self.toolbox.select(self.population, len(self.population))
        offspring = list(map(self.toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                self.toolbox.mate(child1, child2)
        for mutant in offspring:
            if random.random() < 0.2:
                self.toolbox.mutate(mutant)

        # 新しい個体を評価
        self.population = offspring
        for ind in self.population:
            ind.fitness.values = self.toolbox.evaluate(ind)

        # 市場特性を更新
        self.update_market_characteristics()

    def update_market_characteristics(self):
        """
        市場特性を需給、流動性、スリッページを基に動的に更新
        """
        new_characteristics = []

        for agent_actions, alpha, beta, _ in self.market_characteristics:
            # エージェントの行動から需給バランスと取引量を計算
            buy_orders = sum([max(0, action) for action in agent_actions])  # 買い注文量
            sell_orders = sum([abs(min(0, action)) for action in agent_actions])  # 売り注文量

            # 需給バランス D(t)
            supply_and_demand = buy_orders - sell_orders

            # 取引量 V(t)
            volume = buy_orders + sell_orders
            # 流動性の更新
            # 流動性は需給バランスの安定性に応じて増減
            updated_liquidity = alpha * volume - beta * abs(supply_and_demand)

            # スリッページの更新
            # スリッページは需給の変動と流動性に依存
            updated_slippage = abs(supply_and_demand) / (updated_liquidity + 1e-6)

            # 需給の更新（外部要因を含むランダムな変動）
            updated_sd = supply_and_demand + random.uniform(-0.05, 0.05) - 0.02 * updated_slippage

            new_characteristics.append((updated_sd, updated_liquidity, updated_slippage))

        self.market_characteristics = new_characteristics



    def generate_market(self):
        """
        最適な市場モデルを生成
        """
        best_ind = tools.selBest(self.population, 1)[0]
        return gp.compile(best_ind, self.pset)





# 強化学習エージェント
class Agent:
    def __init__(self, input_dim):
        self.model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_dim=input_dim),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='linear')  # 行動価値
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def train(self, market_dynamics, epochs=10):
        for _ in range(epochs):
            with tf.GradientTape() as tape:
                demands = np.random.uniform(-1, 1, (100, 1))
                liquidities = np.random.uniform(0.1, 1, (100, 1))
                inputs = np.hstack([demands, liquidities])
                rewards = market_dynamics(demands, liquidities)
                predictions = self.model(inputs, training=True)
                loss = tf.reduce_mean(tf.square(predictions - rewards))
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def evaluate(self, market_dynamics):
        demands = np.random.uniform(-1, 1, (100, 1))
        liquidities = np.random.uniform(0.1, 1, (100, 1))
        inputs = np.hstack([demands, liquidities])
        predictions = self.model(inputs, training=False)
        rewards = market_dynamics(demands, liquidities)
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
num_agent = 5
market_gen = MarketGenerator()
agents = [Agent(input_dim=2) for _ in range(num_agent)]
meta_learner = MetaLearner()

for generation in range(10):
    market_dynamics = market_gen.generate_market()
    for agent in agents:
        agent.train(market_dynamics)
        performance = agent.evaluate(market_dynamics)
        meta_learner.store_experience(market_dynamics, performance)

    print(f"Generation {generation}, Best Market Dynamics Evaluated")
    market_gen.evolve()

meta_learner.analyze()
