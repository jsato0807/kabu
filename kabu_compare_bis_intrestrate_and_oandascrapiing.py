# 前述のscrape_from_oanda関数をインポートして使用することを前提にしている

from kabu_oanda_swapscraping import scrape_from_oanda
from kabu_bis_intrestrate import filter_country_data
import pandas as pd

def currency_code_to_country_name(currency_code):
    # 略称と正式名称の辞書
    currency_map = {
        'USD': 'United States',
        'JPY': 'Japan',
        'EUR': 'Eurozone',
        'GBP': 'United Kingdom',
        'AUD': 'Australia',
        'CAD': 'Canada',
        'CHF': 'Switzerland',
        'NZD': 'New Zealand',
        'CNY': 'China',
        'HKD': 'Hong Kong',
        'SGD': 'Singapore',
        'SEK': 'Sweden',
        'NOK': 'Norway',
        'MXN': 'Mexico',
        'BRL': 'Brazil',
        'ZAR': 'South Africa'
    }
    
    return currency_map.get(currency_code, 'Unknown Country')

# 使用例
print(currency_code_to_country_name('USD'))  # America
print(currency_code_to_country_name('JPY'))  # Japan
print(currency_code_to_country_name('EUR'))  # Eurozone


def calculate_swap_averages(pair, start_date, end_date):
    # scrape_from_oanda関数を使って、指定した期間のデータを取得
    swap_data = scrape_from_oanda(pair, start_date, end_date)
    
    # 売りスワップと買いスワップを別々に集計する変数
    total_sell_swap = 0  # 売りスワップの合計
    total_buy_swap = 0   # 買いスワップの合計
    total_days = 0       # 付与日数の合計
    
    # データを処理してスワップを集計
    for row in swap_data:
        try:
            sell_swap = float(row[0]) if row[0] else 0  # 売りスワップ
            buy_swap = float(row[1]) if row[1] else 0   # 買いスワップ
            days = int(row[2])  # 付与日数
            
            if days > 0:
                # スワップと日数を集計
                total_buy_swap += buy_swap
                total_sell_swap += sell_swap
                total_days += days
            
        except (ValueError, IndexError) as e:
            # データが正しくない場合、スキップ
            print(f"Error processing row {row}: {e}")
    
    # 売りスワップと買いスワップの平均を計算
    if total_days > 0:
        average_buy_swap = total_buy_swap / total_days
        average_sell_swap = total_sell_swap / total_days
    else:
        average_buy_swap = 0
        average_sell_swap = 0
    
    return average_buy_swap, average_sell_swap

# 使用例
pair = "USD/JPY"
pair_splits = pair.split("/")

if pair == "ZAR/JPY" or pair == "HKD/JPY":  #rf) https://www.oanda.jp/course/ny4/swap
    order_size = 100000

else:
    order_size = 10000


start_date = "2021-01-01"
end_date = "2021-03-31"

average_buy, average_sell = calculate_swap_averages(pair, start_date, end_date)
print(f"期間: {start_date} から {end_date} の平均買いスワップ: {average_buy}")
print(f"期間: {start_date} から {end_date} の平均売りスワップ: {average_sell}")

AVERAGES = []
for pair in pair_splits:
    interest_rate = filter_country_data(currency_code_to_country_name(pair),pd.to_datetime(start_date),pd.to_datetime(end_date))
    print(interest_rate.head())
    interest_rate = interest_rate['OBS_VALUE:Observation Value']
    print(interest_rate.head())
    average = interest_rate.mean()
    print(average)
    AVERAGES.append(average)

theory = (AVERAGES[0] - AVERAGES[1]) * order_size/100 * 1/365

print(f"theory: {theory}")


