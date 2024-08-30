import yfinance as yf
import pandas as pd
import numpy as np
import talib as ta
from itertools import combinations

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

def download_data(pair, start_date='2019-05-01', end_date='2024-08-01'):
    """
    指定された通貨ペアのデータをyfinanceからダウンロードする関数。
    """
    try:
        # 通貨ペアのデータを取得
        data = yf.download(pair, start=start_date, end=end_date)
        return data
    except Exception as e:
        print(f"Error downloading data for {pair}: {e}")
        return pd.DataFrame()

def calculate_bollinger_bands(data, window=20):
    data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = ta.BBANDS(data['Close'], timeperiod=window)
    return data

def calculate_rsi(data, window=14):
    data['RSI'] = ta.RSI(data['Close'], timeperiod=window)
    return data

def calculate_stochastic(data, window=14):
    data['Stoch_K'], data['Stoch_D'] = ta.STOCH(data['High'], data['Low'], data['Close'], fastk_period=window)
    return data

def calculate_adx(data, window=14):
    data['ADX'] = ta.ADX(data['High'], data['Low'], data['Close'], timeperiod=window)
    return data

def calculate_atr(data, window=14):
    data['ATR'] = ta.ATR(data['High'], data['Low'], data['Close'], timeperiod=window)
    return data

def is_range_market(data):
    """
    レンジ相場の判別を行う関数
    """
    conditions = [
        (data['Close'] > data['BB_Lower']) & (data['Close'] < data['BB_Upper']),
        (data['RSI'] > 30) & (data['RSI'] < 70),
        (data['Stoch_K'] > 20) & (data['Stoch_K'] < 80),
        (data['ADX'] < 20),
        (data['ATR'] < data['ATR'].mean())
    ]
    data['Range_Market'] = np.all(conditions, axis=0)
    return data

def calculate_range_market_period(data):
    """
    レンジ相場の期間の合計を計算する関数
    """
    data = calculate_bollinger_bands(data)
    data = calculate_rsi(data)
    data = calculate_stochastic(data)
    data = calculate_adx(data)
    data = calculate_atr(data)
    data = is_range_market(data)
    return data['Range_Market'].sum()

def organize_data_by_pair(pairs):
    data_dict = {}
    for pair in pairs:
        pair_data = download_data(pair)
        if not pair_data.empty:
            range_period = calculate_range_market_period(pair_data)
            data_dict[pair] = {
                'Open': pair_data['Open'],
                'High': pair_data['High'],
                'Low': pair_data['Low'],
                'Close': pair_data['Close'],
                'Volume': pair_data['Volume'],
                'Adj Close': pair_data['Adj Close'],
                'Range_Period': range_period
            }
    return data_dict

def convert_all_to_pips(data_dict):
    pips_dict = {}
    for pair, data in data_dict.items():
        if pair.endswith('JPY=X'):
            pips_dict[pair] = {
                'Open': data['Open'] * 100,
                'High': data['High'] * 100,
                'Low': data['Low'] * 100,
                'Close': data['Close'] * 100,
                'Volume': data['Volume'],
                'Adj Close': data['Adj Close'] * 100,
                'Range_Period': data['Range_Period']
            }
        else:
            pips_dict[pair] = {
                'Open': data['Open'] * 10000,
                'High': data['High'] * 10000,
                'Low': data['Low'] * 10000,
                'Close': data['Close'] * 10000,
                'Volume': data['Volume'],
                'Adj Close': data['Adj Close'] * 10000,
                'Range_Period': data['Range_Period']
            }
    return pips_dict

def calculate_volatility(data_dict, pairs):
    combined = pd.DataFrame()
    for pair in pairs:
        combined[pair] = data_dict[pair]['Adj Close']
    return combined.sum(axis=1).std()

def calculate_individual_volatilities(data_dict, pairs):
    return sum(data_dict[pair]['Adj Close'].std() for pair in pairs)

def find_minimum_details(data_dict, pairs):
    min_details = {}
    for pair in pairs:
        combined = data_dict[pair]['Adj Close']
        min_value = combined.min()
        min_date = combined.idxmin()
        min_details[pair] = (min_value, min_date)
    
    min_dates = [details[1] for details in min_details.values()]
    min_date = max(set(min_dates), key=min_dates.count)  # 最も頻繁に出現する日付を選択
    
    return min_date, min_details

def check_valid_combination(pairs):
    currencies_in_pairs = [pair[:3] + pair[3:6] for pair in pairs]
    currencies_set = set()
    
    for pair in pairs:
        currencies_set.update({pair[:3], pair[3:6]})
    
    currency_counts = {currency: sum([pair.count(currency) for pair in pairs]) for currency in currencies_set}
    common_currencies = [currency for currency, count in currency_counts.items() if count > 1]
    
    return len(common_currencies) == 2

def display_combination_details(data_dict, pairs):
    combined_volatility = calculate_volatility(data_dict, pairs)
    individual_volatility = calculate_individual_volatilities(data_dict, pairs)
    min_date, min_details = find_minimum_details(data_dict, pairs)
    
    # レンジ相場の期間の合計を計算
    range_period_sum = sum(data_dict[pair]['Range_Period'] for pair in pairs)
    
    print("通貨ペアの組み合わせ:")
    print(f"  組み合わせ: {pairs}")
    print(f"  総計ボラティリティ: {combined_volatility:.4f} pips")
    print(f"  個々の通貨ペアボラティリティの合計: {individual_volatility:.4f} pips")
    print(f"  最小値が発生した日付: {min_date.strftime('%Y-%m-%d')}")
    for pair, (min_value, date) in min_details.items():
        print(f"    {pair} - 最小値: {min_value:.4f} pips - 日付: {date.strftime('%Y-%m-%d')}")
    print(f"  レンジ相場だった期間の合計: {range_period_sum} 日")
    print()


def display_specific_combo_details(data_dict, pairs):
    """
    特定の通貨ペアの組に対して、レンジ相場の期間の合計などの詳細を表示する関数
    """
    if not check_valid_combination(pairs):
        print("無効な通貨ペアの組み合わせです。")
        return

    combined_volatility = calculate_volatility(data_dict, pairs)
    individual_volatility = calculate_individual_volatilities(data_dict, pairs)
    min_date, min_details = find_minimum_details(data_dict, pairs)
    
    # レンジ相場だった期間の合計を計算
    range_period_sum = sum(data_dict[pair]['Range_Period'] for pair in pairs)
    
    print("指定された通貨ペアの組み合わせ:")
    print(f"  組み合わせ: {pairs}")
    print(f"  総計ボラティリティ: {combined_volatility:.4f} pips")
    print(f"  個々の通貨ペアボラティリティの合計: {individual_volatility:.4f} pips")
    print(f"  最小値が発生した日付: {min_date.strftime('%Y-%m-%d')}")
    for pair, (min_value, date) in min_details.items():
        print(f"    {pair} - 最小値: {min_value:.4f} pips - 日付: {date.strftime('%Y-%m-%d')}")
    print(f"  レンジ相場だった期間の合計: {range_period_sum} 日")
    print()



if __name__ == "__main__":
    currencies = ['AUD', 'NZD', 'USD', 'CHF', 'GBP', 'EUR', 'CAD', 'JPY']
    currency_pairs = generate_currency_pairs(currencies)

    data_dict = organize_data_by_pair(currency_pairs)
    data_pips = convert_all_to_pips(data_dict)
    combos = list(combinations(currency_pairs, 3))

    results = []

    for combo in combos:
        try:
            if check_valid_combination(combo):
                combined_volatility = calculate_volatility(data_pips, combo)
                individual_volatility = calculate_individual_volatilities(data_pips, combo)
                min_date, min_details = find_minimum_details(data_pips, combo)
                results.append((combo, combined_volatility, individual_volatility, min_date, min_details))
        except KeyError:
            continue

    results_sorted = sorted(results, key=lambda x: x[1])
    top_10_combos = results_sorted[:10]

    print("総計のボラティリティが小さい順に上位10個の通貨ペアの組み合わせ:")
    for combo, combined_volatility, individual_volatility, min_date, min_details in top_10_combos:
        display_combination_details(data_pips, combo)

        # 任意の3つの通貨ペアの組み合わせを指定して表示
    specific_combos = [('AUDNZD=X', 'AUDUSD=X', 'USDCAD=X'),
                       ('AUDNZD=X', 'AUDCAD=X', 'USDCAD=X'),
                       ('AUDNZD=X', 'NZDUSD=X', 'USDCAD=X'),
                       ('AUDNZD=X', 'NZDCAD=X', 'USDCAD=X'),
                       ('AUDNZD=X', 'AUDUSD=X', 'USDCHF=X'),
                       ]  # 例: 任意の3つの通貨ペア
    for combo in specific_combos:
        display_specific_combo_details(data_pips, combo)
