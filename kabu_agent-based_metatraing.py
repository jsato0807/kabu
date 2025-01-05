import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
from deap import base, creator, tools, gp

# 市場の値動きを生成する遺伝的プログラミング
class MarketGenerator:
    def __init__(self):
        # 遺伝的プログラミングの初期設定
        self.pset = gp.PrimitiveSet("MAIN", 2)  # 需給と流動性を入力
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
        self.toolbox.register("evaluate", self.evaluate_individual)

        self.population = self.toolbox.population(n=10)

    def evaluate_individual(self, individual):
        # 適応度関数: 生成された市場モデルのボラティリティ
        func = gp.compile(individual, self.pset)
        data = [(random.uniform(-1, 1), random.uniform(0.1, 1)) for _ in range(100)]
        price_changes = [func(demand, liquidity) for demand, liquidity in data]
        volatility = np.std(price_changes)
        return volatility,

    def evolve(self):
        # GPによる進化
        tools.Statistics(lambda ind: ind.fitness.values)
        self.population = tools.selBest(self.population, 5) + tools.selTournament(self.population, 5, tournsize=3)
        offspring = self.toolbox.select(self.population, len(self.population))
        offspring = list(map(self.toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                self.toolbox.mate(child1, child2)
        for mutant in offspring:
            if random.random() < 0.2:
                self.toolbox.mutate(mutant)

        self.population = offspring
        for ind in self.population:
            ind.fitness.values = self.toolbox.evaluate(ind)

    def generate_market(self):
        # 最適な市場モデルを生成
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
market_gen = MarketGenerator()
agents = [Agent(input_dim=2) for _ in range(5)]
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
