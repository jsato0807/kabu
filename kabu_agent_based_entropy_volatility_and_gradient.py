import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy, pearsonr

# ----- サンプルデータ（実データに置き換える） -----
np.random.seed(42)
T = 1000  # 期間
discriminator_outputs = np.random.dirichlet(alpha=[1,1,1,1], size=T)  # Discriminator の出力確率
roi_values = np.cumsum(np.random.randn(T) * 0.01)  # ROI（ランダムな増減, 実際は取引データを使用）

# ----- 1. Discriminator のエントロピー H_D を計算 -----
H_D_series = np.array([entropy(discriminator_outputs[t]) for t in range(T)])  # 各時点のエントロピー

# ----- 2. エントロピーの変化率（勾配）を計算 -----
entropy_gradient = np.gradient(H_D_series)

# ----- 3. エントロピーの分散（ボラティリティ）を計算 -----
W = 20  # ウィンドウサイズ
H_D_volatility = np.array([np.var(H_D_series[max(0, t-W):t+1]) for t in range(T)])

# ----- 4. ROI との相関を計算 -----
rho_H_ROI, _ = pearsonr(H_D_volatility, roi_values)  # エントロピーボラティリティとROIの相関

# ----- 5. データフレーム化 -----
df = pd.DataFrame({
    "Time": np.arange(T),
    "Entropy": H_D_series,
    "Entropy Gradient": entropy_gradient,
    "Entropy Volatility": H_D_volatility,
    "ROI": roi_values
})

import ace_tools as tools
tools.display_dataframe_to_user(name="Entropy Analysis", dataframe=df)

print(f"ρ(H_D volatility, ROI): {rho_H_ROI:.4f}")

# ----- 6. グラフ描画 -----
plt.figure(figsize=(12, 8))

# 1. エントロピーの時系列プロット
plt.subplot(3, 1, 1)
plt.plot(df["Time"], df["Entropy"], label="Entropy", color='blue')
plt.xlabel("Time")
plt.ylabel("Entropy")
plt.title("Discriminator Entropy Over Time")
plt.legend()

# 2. エントロピーの変化率（勾配）のプロット
plt.subplot(3, 1, 2)
plt.plot(df["Time"], df["Entropy Gradient"], label="Entropy Gradient", color='purple')
plt.axhline(0, color='black', linestyle='dotted')
plt.xlabel("Time")
plt.ylabel("Entropy Gradient")
plt.title("Entropy Change Rate (Gradient)")
plt.legend()

# 3. エントロピーボラティリティ vs ROI
plt.subplot(3, 1, 3)
plt.scatter(df["Entropy Volatility"], df["ROI"], alpha=0.7, color='green')
plt.xlabel("Entropy Volatility")
plt.ylabel("ROI")
plt.title("Entropy Volatility vs. ROI")

plt.tight_layout()
plt.show()
