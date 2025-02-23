import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy

# ----- サンプルデータ（実データに置き換える） -----
np.random.seed(42)
T = 1000  # 期間
discriminator_outputs = np.random.dirichlet(alpha=[1,1,1,1], size=T)  # Discriminator の出力確率

# ----- 1. Discriminator のエントロピー H_D を計算 -----
H_D_series = np.array([entropy(discriminator_outputs[t]) for t in range(T)])  # 各時点のエントロピー

# ----- 2. エントロピーの移動分散（エントロピーボラティリティ）を計算 -----
W = 20  # ウィンドウサイズ
H_D_volatility = np.array([np.var(H_D_series[max(0, t-W):t+1]) for t in range(T)])  # 移動分散

# ----- 3. データフレーム化 -----
df = pd.DataFrame({
    "Time": np.arange(T),
    "Entropy": H_D_series,
    "Entropy Volatility": H_D_volatility
})

import ace_tools as tools
tools.display_dataframe_to_user(name="Entropy & Volatility", dataframe=df)

# ----- 4. エントロピーのプロット -----
plt.figure(figsize=(12, 6))

# 1. エントロピーの時系列プロット
plt.subplot(2, 1, 1)
plt.plot(df["Time"], df["Entropy"], label="Entropy", color='b')
plt.xlabel("Time")
plt.ylabel("Entropy")
plt.title("Discriminator Entropy Over Time")
plt.legend()

# 2. エントロピーボラティリティ（移動分散）のプロット
plt.subplot(2, 1, 2)
plt.plot(df["Time"], df["Entropy Volatility"], label="Entropy Volatility", color='r')
plt.xlabel("Time")
plt.ylabel("Entropy Volatility")
plt.title("Entropy Volatility Over Time")
plt.legend()

plt.tight_layout()
plt.show()
