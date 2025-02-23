import numpy as np
import matplotlib.pyplot as plt

# --- 1. エントロピーの計算 ---
def compute_entropy(probabilities):
    """
    各時点での出力確率からエントロピーを計算する関数
    probabilities: numpy array of shape (T, num_classes)
    """
    eps = 1e-12  # 数値計算上のゼロ除算を避けるための小さな値
    # エントロピー H = -sum(p_i * log(p_i))
    entropies = -np.sum(probabilities * np.log(probabilities + eps), axis=1)
    return entropies

# 例として、1000時点の4クラス確率データ（Discriminator の出力）を生成
T = 1000
# ここでは、Dirichlet 分布から乱数で生成（実際は学習中の出力確率を利用）
#alpha = np.array([1, 1, 1, 1])
#probabilities = np.random.dirichlet(alpha, size=T)

# 各時点のエントロピーを計算
entropies = compute_entropy(probabilities)

# エントロピーの分散を計算
entropy_variance = np.var(entropies)
print("Entropy Variance: ", entropy_variance)

# エントロピーの推移をプロットして確認
plt.figure(figsize=(10, 4))
plt.plot(entropies, label="Entropy")
plt.xlabel("Time Step")
plt.ylabel("Entropy")
plt.title("Time Series of Discriminator Entropy")
plt.legend()
plt.show()

# --- 2. ROI の計算 ---
def compute_roi(total_assets):
    """
    総資産の時系列データから ROI を計算する関数
    total_assets: numpy array of shape (T,)
    ROI = (最終資産 - 初期資産) / 初期資産
    """
    roi = (total_assets[-1] - total_assets[0]) / total_assets[0]
    return roi

# 例として、1000時点の総資産データを生成
# 初期資産を1000、最終資産を1500程度にノイズを加えながら線形に増加する例
#total_assets = np.linspace(1000, 1500, T) + np.random.normal(0, 10, T)

roi = compute_roi(total_assets)
print("ROI: ", roi)

# 総資産の推移をプロットして確認
plt.figure(figsize=(10, 4))
plt.plot(total_assets, label="Total Assets", color="green")
plt.xlabel("Time Step")
plt.ylabel("Total Assets")
plt.title("Time Series of Discriminator's Total Assets")
plt.legend()
plt.show()
