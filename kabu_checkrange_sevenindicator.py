import yfinance as yf
import numpy as np
import pandas as pd
import talib as ta
import matplotlib.pyplot as plt

# 為替データをダウンロードする関数
def download_forex_data(ticker, period='1y'):
    data = yf.download(ticker, period=period)
    return data

# ボリンジャーバンドを計算
def calculate_bollinger_bands(data, window=20):
    data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = ta.BBANDS(data['Close'], timeperiod=window)
    return data

# RSIを計算
def calculate_rsi(data, window=14):
    data['RSI'] = ta.RSI(data['Close'], timeperiod=window)
    return data

# ストキャスティクスを計算
def calculate_stochastic(data, window=14):
    data['Stoch_K'], data['Stoch_D'] = ta.STOCH(data['High'], data['Low'], data['Close'], fastk_period=window)
    return data

# ADXを計算
def calculate_adx(data, window=14):
    data['ADX'] = ta.ADX(data['High'], data['Low'], data['Close'], timeperiod=window)
    return data

# ATRを計算
def calculate_atr(data, window=14):
    data['ATR'] = ta.ATR(data['High'], data['Low'], data['Close'], timeperiod=window)
    return data

# レンジ相場の判別
def is_range_market(data):
    conditions = [
        (data['Close'] > data['BB_Lower']) & (data['Close'] < data['BB_Upper']),
        (data['RSI'] > 30) & (data['RSI'] < 70),
        (data['Stoch_K'] > 20) & (data['Stoch_K'] < 80),
        (data['ADX'] < 20)
    ]
    data['Range_Market'] = np.all(conditions, axis=0)
    return data

# サポート・レジスタンスラインを追加（単純化のため、過去20期間の最高値と最低値を使用）
def calculate_support_resistance(data, window=20):
    data['Support'] = data['Low'].rolling(window=window).min()
    data['Resistance'] = data['High'].rolling(window=window).max()
    return data

# データの視覚化
def visualize_data(data):
    plt.figure(figsize=(14, 7))

    # 終値のプロット
    plt.plot(data.index, data['Close'], label='Close Price', color='black')

    # ボリンジャーバンドの上限のプロット
    plt.plot(data.index, data['BB_Upper'], label='BB Upper', color='blue', linestyle='--')

    # ボリンジャーバンドの下限のプロット
    plt.plot(data.index, data['BB_Lower'], label='BB Lower', color='blue', linestyle='--')

    # レジスタンスレベルのプロット
    plt.plot(data.index, data['Resistance'], label='Resistance', color='red', linestyle='--')

    # サポートレベルのプロット
    plt.plot(data.index, data['Support'], label='Support', color='green', linestyle='--')

    # レンジ相場のハイライト
    plt.fill_between(data.index, data['Close'].min(), data['Close'].max(),
                     where=data['Range_Market'], color='yellow', alpha=0.3, label='Range Market')

    # ラベルとタイトルの設定
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Price Chart with Indicators')
    plt.legend()

    # グリッドの追加
    plt.grid(True)

    # プロットの表示
    plt.tight_layout()
    plt.show()

# メインの実行部分
def main():
    # 通貨ペアのリスト
    currency_pairs = [
    "USDJPY=X", "EURUSD=X", "GBPUSD=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X",
    "EURJPY=X", "EURGBP=X", "EURAUD=X", "EURCAD=X", "EURCHF=X", "EURNZD=X",
    "GBPJPY=X", "GBPAUD=X", "GBPCAD=X", "GBPCHF=X", "GBPNZD=X",
    "AUDJPY=X", "AUDCAD=X", "AUDCHF=X", "AUDNZD=X",
    "CADJPY=X", "CADCHF=X", "CHFJPY=X",
    "NZDJPY=X", "NZDCHF=X"
    ]

    # 各通貨ペアのデータを取得し、レンジ相場の期間を計算
    range_periods = []

    for pair in currency_pairs:
        print(f"Processing {pair}...")
        data = yf.download(pair, start="2019-06-01", end="2024-08-01")

        # 指標を計算
        data = calculate_bollinger_bands(data)
        data = calculate_rsi(data)
        data = calculate_stochastic(data)
        data = calculate_adx(data)
        data = calculate_atr(data)
        data = calculate_support_resistance(data)

        # レンジ相場の判別
        data = is_range_market(data)

        # レンジ相場の期間を計算
        range_period = data['Range_Market'].sum()  # Trueの数をカウント
        range_periods.append((pair, range_period))

    # レンジ相場の期間が長いものから5つを表示
    range_periods.sort(key=lambda x: x[1], reverse=True)
    top_n_range_pairs = range_periods[:20]

    print("\nTop n currency pairs with the longest range market periods:")
    for pair, period in top_n_range_pairs:
        print(f"{pair}: {period} days")

    ## データの視覚化
    #visualize_data(data)

# メイン関数を実行
if __name__ == "__main__":
    main()
