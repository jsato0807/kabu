import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.signal import welch
from hurst import compute_Hc
from scipy.stats import linregress

# ----- 既存の関数をimportする（実際のコードに組み込む際に利用） -----
def variance_ratio(series, lag=2):
    """分散比 (VR) を計算"""
    diff = np.diff(series)
    var_diff_1 = np.var(diff)
    var_diff_q = np.var(series[lag:] - series[:-lag])
    return (var_diff_q / (lag * var_diff_1))

def compute_psd(series):
    """パワースペクトル密度 (PSD) を計算"""
    freqs, psd_values = welch(series, nperseg=len(series)//8)
    return np.sum(psd_values)  # PSDの総エネルギー

def compute_hurst(series):
    """Hurst 指数を計算"""
    H, _, _ = compute_Hc(series, kind='price', simplified=True)
    return H

def compute_acf(series, lag=1):
    """自己相関関数 (ACF) を計算"""
    return acf(series, nlags=lag, fft=True)[lag]

def compute_fractal_dimension(series, scale_min=2, scale_max=20):
    """フラクタル次元 (Box-Counting) を計算"""
    scales = np.arange(scale_min, scale_max)
    counts = []
    
    for scale in scales:
        step_size = len(series) // scale
        split_series = [series[i*step_size:(i+1)*step_size] for i in range(scale)]
        count = np.mean([np.ptp(s) for s in split_series])  # ptp = peak-to-peak (最大値 - 最小値)
        counts.append(count)
    
    log_scales = np.log(scales)
    log_counts = np.log(counts)
    slope, _, _, _, _ = linregress(log_scales, log_counts)
    return -slope  # フラクタル次元

# ----- 1. 元データを作成 -----
np.random.seed(42)
T = 1000  # 期間
market_data = np.cumsum(np.random.randn(T))  # 市場データ

# ----- 2. ノイズレベルの設定 -----
noise_levels = [0, 0.01, 0.05, 0.1, 0.2]  # ノイズの標準偏差

# ----- 3. ノイズを加えた B 指標を計算 -----
B_results = {level: [] for level in noise_levels}

for noise_level in noise_levels:
    for _ in range(30):  # 各ノイズレベルで 30 回試行
        noisy_data = market_data + np.random.normal(0, noise_level, size=T)  # ノイズを加えた市場データ
        
        # 効率性 & 非効率性の指標を計算
        VR = variance_ratio(noisy_data)
        PSD = compute_psd(noisy_data)
        Hurst = compute_hurst(noisy_data)
        ACF = compute_acf(noisy_data)
        Fractal = compute_fractal_dimension(noisy_data)

        # 6種類の B 指標を計算
        B_1 = VR / Hurst  # VR vs Hurst
        B_2 = VR / ACF    # VR vs ACF
        B_3 = VR / Fractal  # VR vs Fractal Dimension
        B_4 = PSD / Hurst  # PSD vs Hurst
        B_5 = PSD / ACF    # PSD vs ACF
        B_6 = PSD / Fractal  # PSD vs Fractal Dimension
        
        B_results[noise_level].extend([B_1, B_2, B_3, B_4, B_5, B_6])

# ----- 4. B 指標のロバスト性を評価（変動係数 & 箱ひげ図） -----
df_B = pd.DataFrame(B_results)

# 各ノイズレベルでの B 指標の変動係数（標準偏差 / 平均）を計算
coefficient_of_variation = df_B.std() / df_B.mean()

# 結果を表示
import ace_tools as tools
tools.display_dataframe_to_user(name="B Robustness Evaluation", dataframe=df_B)

# ----- 5. 箱ひげ図（Box Plot）をプロット -----
plt.figure(figsize=(10, 6))
df_B.boxplot()
plt.xlabel("Noise Level (Standard Deviation)")
plt.ylabel("B Values")
plt.title("Robustness of B Metrics Under Different Noise Levels")
plt.show()

print("\nCoefficient of Variation for B under Different Noise Levels:")
print(coefficient_of_variation)
