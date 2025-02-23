import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import entropy
from statsmodels.tsa.stattools import acf
from hurst import compute_Hc
from scipy.signal import welch
from kabu_agent_based_balance import variance_ratio, compute_hurst

# ----- サンプルデータ（実データに置き換える） -----
np.random.seed(42)
T = 1000  # 期間
market_data = np.cumsum(np.random.randn(T))  # 市場データ
discriminator_outputs = np.random.dirichlet(alpha=[1,1,1,1], size=T)  # Discriminator の出力確率

# ----- 1. 市場の効率性指標（VR, Hurst）を計算 -----
VR_series = np.array([variance_ratio(market_data[:t+1]) for t in range(10, T)])  # VR の時系列
Hurst_series = np.array([compute_hurst(market_data[:t+1]) for t in range(10, T)])  # Hurst の時系列

# ----- 2. Discriminator のエントロピー H_D を計算 -----
H_D_series = np.array([entropy(discriminator_outputs[t]) for t in range(10, T)])  # エントロピーの時系列

# ----- 3. 相関係数 (Pearson ρ) を計算 -----
rho_H_VR, _ = pearsonr(H_D_series, VR_series)  # ρ(H, VR)
rho_H_Hurst, _ = pearsonr(H_D_series, Hurst_series)  # ρ(H, Hurst)

# ----- 結果を表示 -----
df = pd.DataFrame({
    "Time": np.arange(10, T),
    "H_D (Entropy)": H_D_series,
    "VR (Variance Ratio)": VR_series,
    "Hurst Index": Hurst_series
})

import ace_tools as tools
tools.display_dataframe_to_user(name="Market Efficiency vs Entropy", dataframe=df)

print(f"ρ(H, VR): {rho_H_VR:.4f}")
print(f"ρ(H, Hurst): {rho_H_Hurst:.4f}")
