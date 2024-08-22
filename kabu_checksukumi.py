import yfinance as yf
import pandas as pd
import numpy as np
from itertools import combinations

# 通貨ペアのリストを生成する関数
def generate_currency_pairs(currencies):
    """
    通貨リストから通貨ペアを生成する関数。
    重複する通貨ペア（例: 'CHFEUR' と 'EURCHF'）を除外する。
    """
    pairs = []
    for i in range(len(currencies)):
        for j in range(len(currencies)):
            if i != j:
                pair1 = currencies[i] + currencies[j] + '=X'
                pair2 = currencies[j] + currencies[i] + '=X'
                if pair1 not in pairs and pair2 not in pairs:
                    pairs.append(pair1)
    return pairs

# 通貨ペアのリストを生成
currencies = ['AUD', 'NZD', 'USD', 'CHF', 'GBP', 'EUR', 'CAD', 'JPY']
currency_pairs = generate_currency_pairs(currencies)

# データをダウンロードする関数
def download_data(pairs, start_date='2022-07-01', end_date='2022-11-30'):
    """
    指定された通貨ペアのデータをyfinanceから一括でダウンロードする関数。
    """
    try:
        # 通貨ペアのリストを価格データとして取得
        data = yf.download(pairs, start=start_date, end=end_date)['Adj Close']
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return pd.DataFrame()

# データのダウンロード
data = download_data(currency_pairs)

# 通貨ペアの組み合わせを生成
combos = list(combinations(currency_pairs, 3))

def calculate_volatility(df, pairs):
    """
    指定された通貨ペアのデータフレームに基づいて、総和の値動きを計算する。
    """
    combined = df[list(pairs)].sum(axis=1)
    return combined.std()

def check_valid_combination(pairs):
    """
    通貨ペアの組み合わせが条件に合致しているか確認する関数。
    """
    currencies_in_pairs = [pair[:3] + pair[3:6] for pair in pairs]
    currencies_set = set()
    
    for pair in pairs:
        currencies_set.update({pair[:3], pair[3:6]})
    
    # 2つの通貨が共通しているか確認
    currency_counts = {currency: sum([pair.count(currency) for pair in pairs]) for currency in currencies_set}
    common_currencies = [currency for currency, count in currency_counts.items() if count > 1]
    
    # 共通する通貨の数が2つであること
    return len(common_currencies) == 2

# 最適な組み合わせを見つける
results = []

for combo in combos:
    try:
        if check_valid_combination(combo):
            volatility = calculate_volatility(data, combo)
            results.append((combo, volatility))
    except KeyError:
        # 指定された通貨ペアのデータが不足している場合
        continue

# ボラティリティが小さい順にソートし、上位5つを選択
results_sorted = sorted(results, key=lambda x: x[1])
top_5_combos = results_sorted[:5]

# 結果を表示
print("最小の値動きの上位5つの通貨ペアの組み合わせ（条件に合致するもの）:")
for combo, volatility in top_5_combos:
    print(f"組み合わせ: {combo} - 値動き: {volatility:.4f}")
