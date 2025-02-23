import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.signal import welch
from hurst import compute_Hc
from scipy.stats import linregress

# サンプルデータ（実際のデータに置き換え）
np.random.seed(42)
T = 1000  # 期間
market_data = np.cumsum(np.random.randn(T))  # 市場データ
model_data = np.cumsum(np.random.randn(T))  # モデルの出力

# -------- 1. 分散比 (VR) の計算 --------
def variance_ratio(series, lag=2):
    diff = np.diff(series)
    var_diff_1 = np.var(diff)
    var_diff_q = np.var(series[lag:] - series[:-lag])
    return (var_diff_q / (lag * var_diff_1))

VR_market = variance_ratio(market_data)
VR_model = variance_ratio(model_data)

# -------- 2. パワースペクトル密度 (PSD) の計算 --------
def compute_psd(series):
    freqs, psd_values = welch(series, nperseg=len(series)//8)
    return np.sum(psd_values)  # PSDの総エネルギー

PSD_market = compute_psd(market_data)
PSD_model = compute_psd(model_data)

# -------- 3. Hurst 指数の計算 --------
def compute_hurst(series):
    H, _, _ = compute_Hc(series, kind='price', simplified=True)
    return H

Hurst_market = compute_hurst(market_data)
Hurst_model = compute_hurst(model_data)

# -------- 4. 自己相関関数 (ACF) の計算 --------
def compute_acf(series, lag=1):
    return acf(series, nlags=lag, fft=True)[lag]

ACF_market = compute_acf(market_data)
ACF_model = compute_acf(model_data)

# -------- 5. フラクタル次元 (Box-Counting) の計算 --------
def compute_fractal_dimension(series, scale_min=2, scale_max=20):
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

Fractal_market = compute_fractal_dimension(market_data)
Fractal_model = compute_fractal_dimension(model_data)

# -------- 効率性と非効率性の指標の比率を計算 --------
efficiency_ratio_VR = VR_model / VR_market
efficiency_ratio_PSD = PSD_model / PSD_market

inefficiency_ratio_Hurst = Hurst_model / Hurst_market
inefficiency_ratio_ACF = ACF_model / ACF_market
inefficiency_ratio_Fractal = Fractal_model / Fractal_market

# -------- 6種類の B の計算 --------
B_1 = VR_model / Hurst_model  # VR vs Hurst
B_2 = VR_model / ACF_model    # VR vs ACF
B_3 = VR_model / Fractal_model  # VR vs Fractal Dimension
B_4 = PSD_model / Hurst_model  # PSD vs Hurst
B_5 = PSD_model / ACF_model    # PSD vs ACF
B_6 = PSD_model / Fractal_model  # PSD vs Fractal Dimension

# -------- 結果の表示 --------
df = pd.DataFrame({
    "Metric": ["Variance Ratio (VR)", "Power Spectral Density (PSD)", "Hurst Index", "Autocorrelation (ACF)", "Fractal Dimension"],
    "Market": [VR_market, PSD_market, Hurst_market, ACF_market, Fractal_market],
    "Model": [VR_model, PSD_model, Hurst_model, ACF_model, Fractal_model],
    "Efficiency Ratio (Model/Market)": [efficiency_ratio_VR, efficiency_ratio_PSD, None, None, None],
    "Inefficiency Ratio (Model/Market)": [None, None, inefficiency_ratio_Hurst, inefficiency_ratio_ACF, inefficiency_ratio_Fractal]
})

import ace_tools as tools
tools.display_dataframe_to_user(name="Efficiency & Inefficiency Ratios", dataframe=df)

# B 指標の表示
B_values = pd.DataFrame({
    "B Index": ["B_1 (VR/Hurst)", "B_2 (VR/ACF)", "B_3 (VR/Fractal)", "B_4 (PSD/Hurst)", "B_5 (PSD/ACF)", "B_6 (PSD/Fractal)"],
    "Value": [B_1, B_2, B_3, B_4, B_5, B_6]
})

tools.display_dataframe_to_user(name="B Metrics", dataframe=B_values)
