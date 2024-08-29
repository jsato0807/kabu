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


def download_data(pairs, start_date='2019-05-01', end_date='2024-08-01'):
    try:
        data = yf.download(pairs, start=start_date, end=end_date, group_by='ticker')
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return pd.DataFrame()

def convert_all_to_pips(df):
    pips_df = pd.DataFrame()
    for pair in df.columns.levels[0]:
        close_prices = df[pair]['Close']
        if pair.endswith('JPY=X'):
            pips_df[pair] = close_prices * 100
        else:
            pips_df[pair] = close_prices * 10000
    return pips_df

def calculate_bollinger_bands(data, window=20):
    for pair in data.columns:
        close_prices = data[pair]
        bb_upper, bb_middle, bb_lower = ta.BBANDS(close_prices, timeperiod=window)
        data[pair + '_BB_Upper'] = bb_upper
        data[pair + '_BB_Middle'] = bb_middle
        data[pair + '_BB_Lower'] = bb_lower
    return data

def calculate_rsi(data, window=14):
    for pair in data.columns:
        close_prices = data[pair]
        data[pair + '_RSI'] = ta.RSI(close_prices, timeperiod=window)
    return data

def calculate_stochastic(data, window=14):
    for pair in data.columns:
        high_prices = data[pair]
        low_prices = data[pair]
        close_prices = data[pair]
        stoch_k, stoch_d = ta.STOCH(high_prices, low_prices, close_prices, fastk_period=window)
        data[pair + '_Stoch_K'] = stoch_k
        data[pair + '_Stoch_D'] = stoch_d
    return data

def calculate_adx(data, window=14):
    for pair in data.columns:
        high_prices = data[pair]
        low_prices = data[pair]
        close_prices = data[pair]
        data[pair + '_ADX'] = ta.ADX(high_prices, low_prices, close_prices, timeperiod=window)
    return data

def calculate_atr(data, window=14):
    for pair in data.columns:
        high_prices = data[pair]
        low_prices = data[pair]
        close_prices = data[pair]
        data[pair + '_ATR'] = ta.ATR(high_prices, low_prices, close_prices, timeperiod=window)
    return data

def is_range_market(data):
    conditions = [
        (data[pair + '_Close'] > data[pair + '_BB_Lower']) & (data[pair + '_Close'] < data[pair + '_BB_Upper'])
        for pair in data.columns if '_Close' in pair
    ]
    data['Range_Market'] = np.all(conditions, axis=0)
    return data

def calculate_indicators(data):
    data = calculate_bollinger_bands(data)
    data = calculate_rsi(data)
    data = calculate_stochastic(data)
    data = calculate_adx(data)
    data = calculate_atr(data)
    data = is_range_market(data)
    return data

def calculate_range_market_duration(data):
    return data['Range_Market'].sum()

def calculate_volatility(df, pairs):
    combined = df[list(pairs)].sum(axis=1)
    return combined.std()

def calculate_individual_volatilities(df, pairs):
    return sum(df[pair].std() for pair in pairs)

def find_minimum_details(df, pairs):
    min_details = {}
    for pair in pairs:
        combined = df[pair]
        min_value = combined.min()
        min_date = combined.idxmin()
        min_details[pair] = (min_value, min_date)
    
    min_dates = [details[1] for details in min_details.values()]
    min_date = max(set(min_dates), key=min_dates.count)
    
    return min_date, min_details

def check_valid_combination(pairs):
    currencies_in_pairs = [pair[:3] + pair[3:6] for pair in pairs]
    currencies_set = set()
    
    for pair in pairs:
        currencies_set.update({pair[:3], pair[3:6]})
    
    currency_counts = {currency: sum([pair.count(currency) for pair in pairs]) for currency in currencies_set}
    common_currencies = [currency for currency, count in currency_counts.items() if count > 1]
    
    return len(common_currencies) == 2

def display_combination_details(df, pairs):
    combined_volatility = calculate_volatility(df, pairs)
    individual_volatility = calculate_individual_volatilities(df, pairs)
    min_date, min_details = find_minimum_details(df, pairs)
    range_market_duration = calculate_range_market_duration(df)
    
    result = {
        'Pairs': pairs,
        'Total Volatility': combined_volatility,
        'Individual Volatility': individual_volatility,
        'Min Date': min_date,
        'Range Market Duration': range_market_duration,
        'Min Details': min_details
    }
    
    return result

if __name__ == "__main__":
    currencies = ['AUD', 'NZD', 'USD', 'CHF', 'GBP', 'EUR', 'CAD', 'JPY']
    currency_pairs = generate_currency_pairs(currencies)
    data = download_data(currency_pairs)

    if not data.empty:
        data_pips = convert_all_to_pips(data)
        data_indicators = calculate_indicators(data_pips)
        combos = list(combinations(currency_pairs, 3))

        results = []
        for combo in combos:
            try:
                if check_valid_combination(combo):
                    result = display_combination_details(data_pips, combo)
                    results.append(result)
                    
            except Exception as e:
                print(f"Error processing combination {combo}: {e}")

        # Sort by Total Volatility and display top 20 combinations
        results_sorted = sorted(results, key=lambda x: x['Total Volatility'])
        top_20_results = results_sorted[:20]

        for result in top_20_results:
            pairs = result['Pairs']
            print("通貨ペアの組み合わせ:")
            print(f"  組み合わせ: {pairs}")
            print(f"  総計ボラティリティ: {result['Total Volatility']:.4f} pips")
            print(f"  個々の通貨ペアボラティリティの合計: {result['Individual Volatility']:.4f} pips")
            print(f"  最小値が発生した日付: {result['Min Date'].strftime('%Y-%m-%d')}")
            print(f"  レンジ相場の期間の合計: {result['Range Market Duration']} 日")
            for pair, (min_value, date) in result['Min Details'].items():
                print(f"    {pair} - 最小値: {min_value:.4f} pips - 日付: {date.strftime('%Y-%m-%d')}")
            print()
