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


# データをダウンロードする関数
def download_data(pairs, start_date='2019-05-01', end_date='2024-08-01'):
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

# 通貨ペアのリストを受け取り、全てをpipsに変換する関数
def convert_all_to_pips(df):
    pips_df = pd.DataFrame()
    for pair in df.columns:
        if pair.endswith('JPY=X'):
            pips_df[pair] = df[pair] * 100  # JPYは100倍
        else:
            pips_df[pair] = df[pair] * 10000  # その他の通貨ペアは10000倍
    return pips_df

def calculate_volatility(df, pairs):
    """
    指定された通貨ペアのデータフレームに基づいて、総和の値動きを計算する。
    """
    combined = df[list(pairs)].sum(axis=1)
    return combined.std()

def calculate_individual_volatilities(df, pairs):
    """
    指定された通貨ペアのデータフレームに基づいて、個々の通貨ペアのボラティリティの合計を計算する。
    """
    return sum(df[pair].std() for pair in pairs)

def find_minimum_details(df, pairs):
    """
    指定された通貨ペアのデータフレームに基づいて、各通貨ペアの最小値とその日付を出力する関数。
    """
    min_details = {}
    for pair in pairs:
        combined = df[pair]
        min_value = combined.min()
        min_date = combined.idxmin()
        min_details[pair] = (min_value, min_date)
    
    min_dates = [details[1] for details in min_details.values()]
    min_date = max(set(min_dates), key=min_dates.count)  # 最も頻繁に出現する日付を選択
    
    return min_date, min_details

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

def display_combination_details(df, pairs):
    """
    任意の通貨ペア3つについて、総計ボラティリティ、個々の通貨ペアのボラティリティの合計、
    最小値が発生した日付、各通貨ペアの最小値とその日付を出力する関数。
    """
    combined_volatility = calculate_volatility(df, pairs)
    individual_volatility = calculate_individual_volatilities(df, pairs)
    min_date, min_details = find_minimum_details(df, pairs)
    
    print("通貨ペアの組み合わせ:")
    print(f"  組み合わせ: {pairs}")
    print(f"  総計ボラティリティ: {combined_volatility:.4f} pips")
    print(f"  個々の通貨ペアボラティリティの合計: {individual_volatility:.4f} pips")
    print(f"  最小値が発生した日付: {min_date.strftime('%Y-%m-%d')}")
    for pair, (min_value, date) in min_details.items():
        print(f"    {pair} - 最小値: {min_value:.4f} pips - 日付: {date.strftime('%Y-%m-%d')}")
    print()


if __name__ == "__main__":
    # 通貨ペアのリストを生成
    currencies = ['AUD', 'NZD', 'USD', 'CHF', 'GBP', 'EUR', 'CAD', 'JPY']
    currency_pairs = generate_currency_pairs(currencies)

    # データのダウンロード
    data = download_data(currency_pairs)

    # pips単位でデータを変換
    data_pips = convert_all_to_pips(data)

    # 通貨ペアの組み合わせを生成
    combos = list(combinations(currency_pairs, 3))

    # 結果を格納するリスト
    results = []

    for combo in combos:
        try:
            if check_valid_combination(combo):
                combined_volatility = calculate_volatility(data_pips, combo)
                individual_volatility = calculate_individual_volatilities(data_pips, combo)
                min_date, min_details = find_minimum_details(data_pips, combo)
                results.append((combo, combined_volatility, individual_volatility, min_date, min_details))
        except KeyError:
            # 指定された通貨ペアのデータが不足している場合
            continue

    # ボラティリティが小さい順にソートし、上位10個を選択
    results_sorted = sorted(results, key=lambda x: x[1])
    top_10_combos = results_sorted[:10]

    # 結果を表示
    print("総計のボラティリティが小さい順に上位10個の通貨ペアの組み合わせ:")
    for combo, combined_volatility, individual_volatility, min_date, min_details in top_10_combos:
        print(f"組み合わせ: {combo}")
        print(f"  総計ボラティリティ: {combined_volatility:.4f} pips")
        print(f"  個々の通貨ペアボラティリティの合計: {individual_volatility:.4f} pips")
        print(f"  最小値が発生した日付: {min_date.strftime('%Y-%m-%d')}")
        for pair, (min_value, date) in min_details.items():
            print(f"    {pair} - 最小値: {min_value:.4f} pips - 日付: {date.strftime('%Y-%m-%d')}")
        print()

    display_combination_details(data_pips, ['AUDNZD=X','NZDUSD=X','USDCHF=X'])
    display_combination_details(data_pips, ['AUDNZD=X','NZDCAD=X','USDCAD=X'])
    display_combination_details(data_pips, ['AUDNZD=X','NZDCAD=X','USDCAD=X'])
    display_combination_details(data_pips, ['AUDNZD=X','AUDUSD=X','USDCAD=X'])
    display_combination_details(data_pips, ['AUDNZD=X','AUDCAD=X','USDCAD=X'])