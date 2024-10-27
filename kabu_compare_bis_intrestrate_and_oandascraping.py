from kabu_oanda_swapscraping import scrape_from_oanda
from kabu_bis_intrestrate import filter_country_data
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

# Pandasの表示設定
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)
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
def multiple_period_swap_comparison(pair, start_date, end_date, order_size,months_interval):
    current_start = datetime.strptime(start_date, "%Y-%m-%d")
    final_end = datetime.strptime(end_date, "%Y-%m-%d")
    results = []

    while current_start < final_end:
        current_end = current_start + relativedelta(months=months_interval)
        if current_end > final_end:
            current_end = final_end

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

        results.append({
            "期間": f"{current_start.strftime('%Y-%m-%d')} - {current_end.strftime('%Y-%m-%d')}",
            "平均買いスワップ": avg_buy,
            "平均売りスワップ": avg_sell,
            "理論スワップ": theory,
            "買いスワップ割合": buy_ratio,
            "売りスワップ割合": sell_ratio
        })

        current_start = current_end

    comparison_df = pd.DataFrame(results)
    return comparison_df

# 使用例
pair = "USD/JPY"
start_date = "2019-04-01"
end_date = "2024-09-30"
order_size = 10000 if pair != "ZAR/JPY" and pair != "HKD/JPY" else 100000
months_interval = 1

comparison_df = multiple_period_swap_comparison(pair, start_date, end_date, order_size, months_interval)
print(comparison_df)

# 結果をCSVファイルとして保存
comparison_df.to_csv(f"./kabu_compare_bis_intrestrate_and_oandascraping_{pair}_{start_date}_{end_date}_{months_interval}_results.csv", index=False, encoding='utf-8-sig')

# グラフを作成
plt.figure(figsize=(10, 12))

# スワップポイントグラフ
plt.subplot(2, 1, 1)
plt.plot(comparison_df['期間'], comparison_df['平均買いスワップ'], label='平均買いスワップ', marker='o')
plt.plot(comparison_df['期間'], comparison_df['平均売りスワップ'], label='平均売りスワップ', marker='o')
plt.plot(comparison_df['期間'], comparison_df['理論スワップ'], label='理論スワップ', marker='o')

plt.title(f"{pair}のスワップポイント比較")
plt.xlabel("期間")
plt.ylabel("スワップポイント")
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()

# グラフを保存
plt.savefig(f"./kabu_compare_bis_intrestrate_and_oandascraping_{pair}_{start_date}_{end_date}_{months_interval}_graph.png")

# 割合グラフ
plt.subplot(2, 1, 2)
plt.plot(comparison_df['期間'], comparison_df['買いスワップ割合'], label='買いスワップ割合', marker='o')
plt.plot(comparison_df['期間'], comparison_df['売りスワップ割合'], label='売りスワップ割合', marker='o')

plt.title(f"{pair}のスワップ割合比較")
plt.xlabel("期間")
plt.ylabel("割合")
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()

# 割合グラフを保存
plt.savefig(f"./kabu_compare_bis_intrestrate_and_oandascraping_ratio_{pair}_{start_date}_{end_date}_{months_interval}_graph.png")

# グラフを表示
plt.show()