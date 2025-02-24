import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def rolling_volatility(prices, window=50):
    """ボラティリティ（標準偏差）の時間変化を計算"""
    return np.array([np.std(prices[i-window:i]) if i >= window else np.nan for i in range(len(prices))])

def rolling_autocorrelation(prices, window=50):
    """自己相関の時間変化を計算"""
    return np.array([
        np.corrcoef(prices[i-window:i-1], prices[i-window+1:i])[0, 1] if i >= window else np.nan
        for i in range(len(prices))
    ])

def analyze_current_price(current_price, num_clusters=3, window=50):
    """
    current_price: 価格データのリストまたはNumPy配列
    num_clusters: クラスタリングのクラスタ数（デフォルト3）
    window: 移動窓（ボラティリティ・自己相関の計算用）
    """

    # クラスタリング（K-Means）
    price_array = np.array(current_price).reshape(-1, 1)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(price_array)

    # ボラティリティ（標準偏差）の時間変化
    volatilities = rolling_volatility(current_price, window)

    # 自己相関（ラグ1）の時間変化
    autocorrelations = rolling_autocorrelation(current_price, window)

    # 変化率（リターン）: log return
    returns = np.diff(np.log(current_price))

    # クラスタリング結果の表示
    print(f"クラスタリングの結果（K-Means, {num_clusters}クラスタ）: {np.unique(labels, return_counts=True)}")
    print(f"ボラティリティ（全体平均）: {np.nanmean(volatilities):.6f}")
    print(f"自己相関（全体平均）: {np.nanmean(autocorrelations):.6f}")
    print(f"リターンの統計情報: 平均={np.mean(returns):.6f}, 標準偏差={np.std(returns):.6f}")

    # --- グラフのプロット ---
    fig, axs = plt.subplots(4, 1, figsize=(12, 12))

    # (1) クラスタリング結果の可視化
    axs[0].scatter(range(len(current_price)), current_price, c=labels, cmap='viridis', alpha=0.7)
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Current Price")
    axs[0].set_title("Current Price Clustering (K-Means)")
    axs[0].colorbar(label="Cluster")

    # (2) ボラティリティの時間変化
    axs[1].plot(volatilities, color='blue', label="Volatility")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Volatility")
    axs[1].set_title("Rolling Volatility Over Time")
    axs[1].legend()
    axs[1].grid(True)

    # (3) 自己相関の時間変化
    axs[2].plot(autocorrelations, color='red', label="Autocorrelation (Lag 1)")
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Autocorrelation")
    axs[2].set_title("Rolling Autocorrelation Over Time")
    axs[2].legend()
    axs[2].grid(True)

    # (4) リターンのヒストグラム
    axs[3].hist(returns, bins=50, alpha=0.7, color='blue', edgecolor='black', density=True)
    axs[3].set_xlabel("Log Return")
    axs[3].set_ylabel("Frequency")
    axs[3].set_title("Histogram of Log Returns")
    axs[3].grid(True)

    plt.tight_layout()
    plt.show()

    return {
        "クラスタリングラベル": labels,
        "ボラティリティ": volatilities,
        "自己相関": autocorrelations,
        "リターン": returns
    }
