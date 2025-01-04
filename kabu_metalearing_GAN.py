import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# 市場環境の次元
ENV_DIM = 100
NUM_AGENTS = 3  # エージェント数

# ジェネレーター（市場環境を生成）
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(128, activation="relu", input_dim=10),
        layers.Dense(256, activation="relu"),
        layers.Dense(ENV_DIM, activation="linear")
    ])
    return model

# 識別者（各エージェントが環境を評価）
def build_agent():
    model = tf.keras.Sequential([
        layers.Dense(128, activation="relu", input_dim=ENV_DIM),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    return model

# 複数エージェントの構造
agents = [build_agent() for _ in range(NUM_AGENTS)]
generator = build_generator()

# オプティマイザー
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
discriminator_optimizers = [tf.keras.optimizers.Adam(learning_rate=0.0002) for _ in range(NUM_AGENTS)]

# 損失関数の計算
def compute_discriminator_loss(real_env, fake_env):
    real_scores = [agent(real_env) for agent in agents]
    fake_scores = [agent(fake_env) for agent in agents]
    real_loss = tf.reduce_mean([tf.keras.losses.binary_crossentropy(tf.ones_like(score), score) for score in real_scores])
    fake_loss = tf.reduce_mean([tf.keras.losses.binary_crossentropy(tf.zeros_like(score), score) for score in fake_scores])
    return real_loss + fake_loss

def compute_generator_loss(fake_env):
    fake_scores = [agent(fake_env) for agent in agents]
    fake_loss = tf.reduce_mean([tf.keras.losses.binary_crossentropy(tf.ones_like(score), score) for score in fake_scores])
    return fake_loss

# トレーニングループ
EPOCHS = 5000
BATCH_SIZE = 32
real_data = np.random.randn(1000, ENV_DIM)  # 本物の市場データ

for epoch in range(EPOCHS):
    # 実際の市場データと生成データ
    real_env = real_data[np.random.randint(0, real_data.shape[0], BATCH_SIZE)]
    noise = np.random.randn(BATCH_SIZE, 10)
    fake_env = generator.predict(noise)

    # 識別者（エージェント）のトレーニング
    for i, agent in enumerate(agents):
        with tf.GradientTape() as tape:
            real_score = agent(real_env)
            fake_score = agent(fake_env)
            real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_score), real_score)
            fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_score), fake_score)
            d_loss = tf.reduce_mean(real_loss + fake_loss)
        gradients = tape.gradient(d_loss, agent.trainable_variables)
        discriminator_optimizers[i].apply_gradients(zip(gradients, agent.trainable_variables))

    # 生成者のトレーニング
    with tf.GradientTape() as tape:
        fake_env = generator(noise)
        g_loss = compute_generator_loss(fake_env)
    gradients = tape.gradient(g_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))

    # 進捗の表示
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, D Loss: {d_loss.numpy():.4f}, G Loss: {g_loss.numpy():.4f}")

print("トレーニング完了")

