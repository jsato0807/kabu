import yfinance as yf
import pandas as pd
from itertools import combinations

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

# 通貨ペアのリストを生成
currencies = ['AUD', 'NZD', 'USD', 'CHF', 'GBP', 'EUR', 'CAD', 'JPY']
currency_pairs = generate_currency_pairs(currencies)

# データをダウンロードする関数
def download_data(pairs, start_date='2022-07-01', end_date='2022-11-30'):
    try:
        data = yf.download(pairs, start=start_date, end=end_date)['Adj Close']
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return pd.DataFrame()

# データのダウンロード
data = download_data(currency_pairs)

# 通貨ペアの組み合わせを生成
combos = list(combinations(currency_pairs, 3))

# pipsに変換する関数
def convert_to_pips(df, pair):
    if pair.endswith('JPY=X'):
        return df[pair] * 100
    else:
        return df[pair] * 10000

# 任意の3通貨ペアの総計のボラティリティを計算する関数
def calculate_total_volatility(df, pairs):
    combined = df[list(pairs)].sum(axis=1)
    return combined.std()

def calculate_individual_volatility(df, pair):
    pips_series = convert_to_pips(df, pair)
    return pips_series.std()

# 各通貨ペアの最小値とその時の総計を計算して返す関数
def get_min_value_and_total(df, pair):
    pips_series = convert_to_pips(df, pair)
    min_value = pips_series.min()
    min_index = pips_series.idxmin()
    total_value_on_min_date = df.loc[min_index, list(df.columns)].sum()
    return min_value, total_value_on_min_date

# 組み合わせのボラティリティと最小値を計算してリストで返す関数
def calculate_volatilities_and_min_values(combos, data):
    results = []
    for combo in combos:
        try:
            total_volatility = calculate_total_volatility(data, combo)
            individual_results = []
            for pair in combo:
                min_value, total_on_min = get_min_value_and_total(data, pair)
                indiv_vol = calculate_individual_volatility(data, pair)
                individual_results.append((pair, indiv_vol, min_value, total_on_min))
            results.append((combo, total_volatility, individual_results))
        except KeyError:
            continue
    return sorted(results, key=lambda x: x[1])  # 総計ボラティリティでソート

# 結果の表示
top_10_results = calculate_volatilities_and_min_values(combos, data)[:10]
for combo, total_volatility, individual_results in top_10_results:
    print(f"\n組み合わせ: {combo} - 総計ボラティリティ: {total_volatility:.4f}")
    for pair, indiv_vol, min_value, total_on_min in individual_results:
        print(f"  ペア: {pair} - 個別ボラティリティ: {indiv_vol:.4f} - 最小値: {min_value:.4f} - その時の総計: {total_on_min:.4f}")

# 任意の3通貨ペアの例として、以下の3つのペアのボラティリティ、最小値、総計を計算してみます。
EXAMPLE_PAIRS = [('AUDNZD=X', 'AUDUSD=X', 'USDCAD=X'),('AUDNZD=X', 'AUDCAD=X', 'USDCAD=X'),('AUDNZD=X', 'NZDUSD=X', 'USDCAD=X'),('AUDNZD=X', 'NZDCAD=X', 'USDCAD=X'),('AUDNZD=X', 'USDCHF=X', 'NZDUSD=X')]
for example_pairs in EXAMPLE_PAIRS:
    example_volatility = calculate_total_volatility(data, example_pairs)
    print(f"\n例の3通貨ペアのボラティリティ ({example_pairs}): {example_volatility:.4f}")

    # 例の3通貨ペアの最小値とその時の総計を表示
    print("例の3通貨ペアの最小値とその時の総計:")
    for pair in example_pairs:
        min_value, total_on_min = get_min_value_and_total(data, pair)
        indiv_vol = calculate_individual_volatility(data, pair)
        print(f"  ペア: {pair} - 個別ボラティリティ: {indiv_vol:.4f} - 最小値: {min_value:.4f} - その時の総計: {total_on_min:.4f}")
