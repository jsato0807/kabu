from kabu_oanda_swapscraping import scrape_from_oanda
from kabu_bis_intrestrate import filter_country_data
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Pandasの表示設定
pd.set_option('display.max_columns', None)  # 列数をすべて表示
pd.set_option('display.expand_frame_repr', False)  # 横幅に合わせて折り返し表示しない
pd.set_option('display.max_colwidth', None)  # 各列の表示幅を広げる
# 行数を100に設定（必要に応じて変更）
pd.set_option('display.max_rows', 100)


def currency_code_to_country_name(currency_code):
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

def calculate_swap_averages(pair, start_date, end_date):
    swap_data = scrape_from_oanda(pair, start_date, end_date)
    total_sell_swap, total_buy_swap, total_days = 0, 0, 0
    
    for row in swap_data:
        try:
            sell_swap = float(row[0]) if row[0] else 0
            buy_swap = float(row[1]) if row[1] else 0
            days = int(row[2])
            if days > 0:
                total_buy_swap += buy_swap * days
                total_sell_swap += sell_swap * days
                total_days += days
        except (ValueError, IndexError) as e:
            print(f"Error processing row {row}: {e}")
    
    average_buy_swap = total_buy_swap / total_days if total_days > 0 else 0
    average_sell_swap = total_sell_swap / total_days if total_days > 0 else 0
    
    return average_buy_swap, average_sell_swap

def calculate_theoretical_swap(pair, start_date, end_date, order_size):
    pair_splits = pair.split("/")
    AVERAGES = []
    for currency in pair_splits:
        country_name = currency_code_to_country_name(currency)
        interest_rate = filter_country_data(country_name, pd.to_datetime(start_date), pd.to_datetime(end_date))
        average_rate = interest_rate['OBS_VALUE:Observation Value'].mean()
        AVERAGES.append(average_rate)
    return (AVERAGES[0] - AVERAGES[1]) * order_size / 100 / 365

# 複数期間でのスワップポイント検証
def multiple_period_swap_comparison(pair, start_date, end_date, order_size):
    current_start = datetime.strptime(start_date, "%Y-%m-%d")
    final_end = datetime.strptime(end_date, "%Y-%m-%d")
    results = []

    while current_start < final_end:
        # 現在の期間の終了日を設定（1か月後に設定）
        current_end = current_start + relativedelta(months=1)
        # 最終日を超えないように調整
        if current_end > final_end:
            current_end = final_end

        # スワップの比較を計算
        avg_buy, avg_sell = calculate_swap_averages(pair, current_start.strftime("%Y-%m-%d"), current_end.strftime("%Y-%m-%d"))
        
        pair_splits = pair.split("/")
        AVERAGES = []
        for currency in pair_splits:
            interest_rate = filter_country_data(currency_code_to_country_name(currency), current_start, current_end)
            average_interest = interest_rate['OBS_VALUE:Observation Value'].mean()
            AVERAGES.append(average_interest)
        
        theory = (AVERAGES[0] - AVERAGES[1]) * order_size / 100 * 1 / 365
        buy_ratio = avg_buy / theory if theory != 0 else None
        sell_ratio = avg_sell / theory if theory != 0 else None

        # 結果をリストに追加
        results.append({
            "期間": f"{current_start.strftime('%Y-%m-%d')} - {current_end.strftime('%Y-%m-%d')}",
            "平均買いスワップ": avg_buy,
            "平均売りスワップ": avg_sell,
            "理論スワップ": theory,
            "買いスワップ割合": buy_ratio,
            "売りスワップ割合": sell_ratio
        })

        # 次の期間にスライド
        current_start = current_end

    # 結果をデータフレームに変換
    comparison_df = pd.DataFrame(results)
    return comparison_df

# 使用例
pair = "USD/JPY"
start_date = "2019-04-01"
end_date = "2024-09-30"
order_size = 10000 if pair != "ZAR/JPY" and pair != "HKD/JPY" else 100000

comparison_df = multiple_period_swap_comparison(pair, start_date, end_date, order_size)
print(comparison_df)