import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# 定数の設定
latent_dim = 100  # ランダムノイズの次元
data_dim = 1      # データの次元
batch_size = 64
epochs = 10000
learning_rate = 0.0002

# データの生成 (正規分布に従うデータ)
def generate_real_data(samples=1000):
    x = np.random.normal(0, 1, (samples, data_dim))
    return x.astype(np.float32)

# 生成者 (Generator) モデル
def build_generator():
    model = tf.keras.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(data_dim, activation='linear')  # 出力はデータと同じ次元
    ])
    return model

# 識別者 (Discriminator) モデル
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Input(shape=(data_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # 出力は0～1のスカラー
    ])
    return model

def update_params(real_output, fake_output, real_output_add, fake_output_add):
    """
    real_output_add と fake_output_add を更新し、新しい値を返す
    """
    # 値を直接代入しつつ、計算グラフを維持
    real_output_add = real_output + real_output_add
    fake_output_add = fake_output + fake_output_add
    return real_output_add, fake_output_add



# モデルの作成
generator = build_generator()
discriminator = build_discriminator()

# オプティマイザ
gen_optimizer = tf.keras.optimizers.Adam(learning_rate)
disc_optimizer = tf.keras.optimizers.Adam(learning_rate)

# トレーニングループ
real_data = generate_real_data(samples=10000)

real_output_add = tf.Variable([1000], dtype=tf.float32)
fake_output_add = tf.Variable([10000], dtype=tf.float32)

for epoch in range(epochs):
    # バッチデータの取得
    idx = np.random.randint(0, real_data.shape[0], batch_size)
    real_batch = real_data[idx]
    
    # ランダムノイズの生成
    random_noise = tf.random.normal((batch_size, latent_dim))
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # 生成者の出力（偽データ）
        generated_data = generator(random_noise)
        
        # 識別者の出力
        real_output = discriminator(real_batch)  # 本物のデータ
        fake_output = discriminator(generated_data)  # 偽データ

        real_output_add, fake_output_add = update_params(real_output,fake_output,real_output_add, fake_output_add)
        
        # 損失関数
        gen_loss = -tf.reduce_mean(tf.math.log(fake_output_add + 1e-8))  # 生成者の損失
        disc_loss = -tf.reduce_mean(tf.math.log(real_output_add + 1e-8) + tf.math.log(1 - fake_output_add + 1e-8))  # 識別者の損失
    
    # 勾配の計算
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    # オプティマイザで重みを更新
    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
    
    # ログの表示
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Gen Loss: {gen_loss.numpy():.4f}, Disc Loss: {disc_loss.numpy():.4f}")

# 生成したデータの可視化
random_noise = tf.random.normal((1000, latent_dim))
generated_samples = generator(random_noise).numpy()

plt.hist(real_data, bins=30, alpha=0.6, label='Real Data')
plt.hist(generated_samples, bins=30, alpha=0.6, label='Generated Data')
plt.legend()
plt.show()
