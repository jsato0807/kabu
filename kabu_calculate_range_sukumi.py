import yfinance as yf
import pandas as pd
import numpy as np
import os
from itertools import combinations
from kabu_checkrange_sevenindicator import calculate_adx, calculate_atr, calculate_bollinger_bands, calculate_rsi, calculate_stochastic, calculate_support_resistance

# 為替データをダウンロードする関数
def download_forex_data(ticker, period='1y'):
    data = yf.download(ticker, period=period)
    return data

# 各通貨ペアのデータをダウンロードし、レンジ相場の期間を計算
def calculate_range_periods(currency_pairs):
    range_periods = []
    for pair in currency_pairs:
        data = download_forex_data(pair)
        data = calculate_bollinger_bands(data)
        data = calculate_rsi(data)
        data = calculate_stochastic(data)
        data = calculate_adx(data)
        data = calculate_atr(data)
        data = calculate_support_resistance(data)
        data = is_range_market(data)
        range_period = data['Range_Market'].sum()  # Trueの数をカウント
        range_periods.append((pair, range_period))
    return range_periods

# ボリンジャーバンド、RSI、ストキャスティクス、ADX、ATR、サポート・レジスタンスの計算（前のコードを利用）
# 省略...

# サポート・レジスタンスラインを追加
def calculate_support_resistance(data, window=20):
    data['Support'] = data['Low'].rolling(window=window).min()
    data['Resistance'] = data['High'].rolling(window=window).max()
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

# 通貨ペアのリストを生成する関数
def generate_currency_pairs(currencies):
    pairs = []
    for i in range(len(currencies)):
        for j in range(len(currencies)):
            if i != j:
                pair1 = currencies[i] + currencies[j] + '=X'
                pair2 = currencies[j] + currencies[i] + '=X'
                if pair1 not in pairs and pair2 not in pairs:
                    pairs.append(pair1)
    return pairs

# データをダウンロードする関数
def download_data(pairs, start_date='2020-07-01', end_date='2024-8-01'):
    try:
        data = yf.download(pairs, start=start_date, end=end_date)['Adj Close']
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return pd.DataFrame()

# 通貨ペアの組み合わせを生成
def generate_combinations(currency_pairs):
    return list(combinations(currency_pairs, 3))

def calculate_volatility(df, pairs):
    combined = df[list(pairs)].sum(axis=1)
    return combined.std()

def check_valid_combination(pairs):
    currencies_in_pairs = [pair[:3] + pair[3:6] for pair in pairs]
    currencies_set = set()
    
    for pair in pairs:
        currencies_set.update({pair[:3], pair[3:6]})
    
    currency_counts = {currency: sum([pair.count(currency) for pair in pairs]) for currency in currencies_set}
    common_currencies = [currency for currency, count in currency_counts.items() if count > 1]
    
    return len(common_currencies) == 2

# メインの実行部分
def main():
    # 通貨ペアのリストを生成
    currencies = ['AUD', 'NZD', 'USD', 'CHF', 'GBP', 'EUR', 'CAD', 'JPY']
    currency_pairs = generate_currency_pairs(currencies)

    # 各通貨ペアのレンジ相場の期間を計算
    range_periods = calculate_range_periods(currency_pairs)

    # レンジ相場の期間が長いものを選択
    range_periods.sort(key=lambda x: x[1], reverse=True)
    top_range_pairs = range_periods[:20]

    # 最上位の通貨ペアからすくみ条件を満たすものを選定
    top_currency_pairs = [pair for pair, _ in top_range_pairs]
    combos = generate_combinations(top_currency_pairs)
    
    data = download_data(top_currency_pairs)
    
    results = []
    for combo in combos:
        try:
            if check_valid_combination(combo):
                volatility = calculate_volatility(data, combo)
                results.append((combo, volatility))
        except KeyError:
            continue
    
    # ボラティリティが小さい順にソート
    results_sorted = sorted(results, key=lambda x: x[1])

    # 結果を表示
    print("最小の値動きの上位n個の通貨ペアの組み合わせ（条件に合致するもの）:")
    for combo, volatility in results_sorted[:]:
        print(f"組み合わせ: {combo} - 値動き: {volatility:.4f}")

if __name__ == "__main__":
    main()
