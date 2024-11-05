from kabu_oanda_swapscraping import scrape_from_oanda
from kabu_bis_intrestrate import filter_country_data
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# Pandasの表示設定
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 100)


class ScrapeSwap:
    def __init__(self,pair,start_date,final_end,order_size,months_interval):
        directory = './csv_dir'
        rename_pair = pair.replace("/", "")
        target_start = datetime.strptime(start_date, '%Y-%m-%d')
        target_end = datetime.strptime(final_end, '%Y-%m-%d')
        
        # ファイル検索と条件に合致するファイルの選択
        found_file = None
        for filename in os.listdir(directory):
            if filename.startswith(f'kabu_oanda_swapscraping_{rename_pair}_from'):
                # ファイルの start と end 日付を抽出
                try:
                    file_start = datetime.strptime(filename.split('_from')[1].split('_to')[0], '%Y-%m-%d')
                    file_end = datetime.strptime(filename.split('_to')[1].split('.csv')[0], '%Y-%m-%d')
                    
                    # start_date と final_end がファイルの範囲内か確認
                    if file_start <= target_start and file_end >= target_end:
                        found_file = filename
                        break
                except ValueError:
                    continue  # 日付フォーマットが違うファイルは無視

        # ファイルを読み込みまたはスクレイピング
        if found_file:
            print(f"Loading data from {found_file}")
            file_path = os.path.join(directory, found_file)
            swap_data = pd.read_csv(file_path)
            swap_data = swap_data.set_index('date').T.to_dict()
        else:
            print(f"scrape_from_oanda({pair}, {start_date}, {final_end})")
            swap_data = scrape_from_oanda(pair, start_date, final_end)

        self.swap_data = swap_data

        self.start_date = start_date
        self.final_end = final_end
        self.pair = pair
        self.order_size = order_size
        self.months_interval = months_interval


        pair_splits = pair.split("/")
        interest_rates = {}
        for currency in pair_splits:
            interest_rate = self.download_interest_rate(currency)
            interest_rates[currency] = interest_rate

        self.interest_rates = interest_rates

    def download_interest_rate(self, currency):
        country_name = self.currency_code_to_country_name(currency)
        directory = './csv_dir'
        target_start = datetime.strptime(self.start_date, '%Y-%m-%d').date()
        target_end = datetime.strptime(self.final_end, '%Y-%m-%d').date()


        # 条件に合致するファイルの検索
        found_file = None
        for filename in os.listdir(directory):
            if filename.startswith(f"kabu_bis_intrestrate_{country_name}_from"):
                try:
                    file_start = datetime.strptime(filename.split('_from')[1].split('_to')[0], '%Y-%m-%d').date()
                    file_end = datetime.strptime(filename.split('_to')[1].split('.csv')[0], '%Y-%m-%d').date()
                    
                    # 指定範囲を包含するファイルがあるか確認
                    if file_start <= target_start and file_end >= target_end:
                        found_file = filename
                        break
                except ValueError:
                    continue  # ファイルのフォーマットが異なる場合はスキップ

        # データ読み込みまたは生成
        if found_file:
            print(f"Loading data from {found_file}")
            file_path = os.path.join(directory, found_file)
            interest_rate = pd.read_csv(file_path)
        else:
            print(f"filter_country_data({country_name}, {target_start}, {target_end})")
            interest_rate = filter_country_data(country_name, pd.to_datetime(target_start), pd.to_datetime(target_end))

        return interest_rate


    def currency_code_to_country_name(self,currency_code):
        currency_map = {
            'USD': 'United_States',
            'JPY': 'Japan',
            'EUR': 'Eurozone',
            'GBP': 'United_Kingdom',
            'AUD': 'Australia',
            'CAD': 'Canada',
            'CHF': 'Switzerland',
            'NZD': 'New_Zealand',
            'CNY': 'China',
            'HKD': 'Hong_Kong',
            'SGD': 'Singapore',
            'SEK': 'Sweden',
            'NOK': 'Norway',
            'MXN': 'Mexico',
            'BRL': 'Brazil',
            'ZAR': 'South_Africa',
            'TRY': 'Turkey'
        }
        return currency_map.get(currency_code, 'Unknown Country')

    def get_data_range(self, data, current_start, current_end):
        result = {}
        start_collecting = False
        for date, values in data.items():            
            # 指定された開始日からデータの収集を開始
            if date == current_start:
                start_collecting = True
            if start_collecting:
                result[date] = values
            # 指定された終了日でループを終了
            if date == current_end:
                break
        return result

    def calculate_swap_averages(self, current_start, current_end):

        total_sell_swap, total_buy_swap, total_days = 0, 0, 0

        filtered_data = self.get_data_range(self.swap_data,current_start,current_end)


        for date, values in filtered_data.items():
            try:
                #total_buy_swap += buy_swap * days
                total_buy_swap += values['buy']
                #total_sell_swap += sell_swap * days
                total_sell_swap += values['sell']
                total_days += values['number_of_days']

            except (ValueError, IndexError) as e:
                print(f"Error processing row, {date}: {values}: {e}")

        average_buy_swap = total_buy_swap / total_days if total_days > 0 else 0
        average_sell_swap = total_sell_swap / total_days if total_days > 0 else 0

        return average_buy_swap, average_sell_swap

    def multiple_period_swap_comparison(self):
    # 複数periodでのスワップポイント検証
        current_start = datetime.strptime(self.start_date, "%Y-%m-%d")
        final_end = datetime.strptime(self.final_end, "%Y-%m-%d")
        results = []
        while current_start < final_end:
            current_end = current_start + relativedelta(months=months_interval) + relativedelta(days=-1)

            if current_end > final_end:
                current_end = final_end

            avg_buy, avg_sell = self.calculate_swap_averages(current_start.strftime("%Y-%m-%d"),current_end.strftime("%Y-%m-%d"))


            pair_splits = pair.split("/")
            AVERAGES = []
            for currency in pair_splits:
                interest_rate = self.interest_rates[currency]
                interest_rate['TIME_PERIOD:Time period or range'] = pd.to_datetime(interest_rate['TIME_PERIOD:Time period or range'])

                current_end_str = current_end.strftime('%Y-%m-%d')

                filtered_interest_rate = interest_rate[interest_rate['TIME_PERIOD:Time period or range'] <= current_end_str]
                average_interest = filtered_interest_rate['OBS_VALUE:Observation Value'].mean()
                AVERAGES.append(average_interest)

            theory = (AVERAGES[0] - AVERAGES[1]) * order_size / 100 * 1 / 365
            buy_ratio = avg_buy / theory if theory != 0 else None
            sell_ratio = avg_sell / theory if theory != 0 else None
            results.append({
                "period": current_start,  # ここをdatetimeオブジェクトに変更

                "average_buy_swap": avg_buy,
                "average_sell_swap": avg_sell,
                "theory swap": theory,
                "buy_swap_ratio": buy_ratio,
                "sell_swap_ratio": sell_ratio
            })
            current_start = current_end + relativedelta(days=1)

        comparison_df = pd.DataFrame(results)
        return comparison_df



# 使用例
pair = "USD/JPY"
start_date = "2019-04-01"
end_date = "2024-10-31"
order_size = 10000 if pair != "ZAR/JPY" and pair != "HKD/JPY" else 100000
months_interval = 1

scrapeswap = ScrapeSwap(pair,start_date,end_date, order_size, months_interval)

comparison_df = scrapeswap.multiple_period_swap_comparison()
print(comparison_df)

pair = pair.replace("/","")
# 結果をCSVファイルとして保存
comparison_df.to_csv(f"./csv_dir/kabu_compare_bis_intrestrate_and_oandascraping_{pair}_{start_date}_{end_date}_{months_interval}_results.csv", index=False, encoding='utf-8-sig')
#print(min(comparison_df['average_buy_swap'].min(), comparison_df['average_sell_swap'].min()))
#print(max(comparison_df['average_buy_swap'].max(), comparison_df['average_sell_swap'].max()))
#exit()

# グラフを作成
plt.figure(figsize=(10, 12))

# スワップポイントグラフ
plt.subplot(2, 1, 1)
ax1 = plt.gca()  # 現在のAxesを取得
ax1.plot(comparison_df['period'], comparison_df['average_buy_swap'], label='average_buy_swap', marker='o')
ax1.plot(comparison_df['period'], comparison_df['average_sell_swap'], label='average_sell_swap', marker='o')

# 理論値をプロットするための新しいy軸を作成
ax2 = ax1.twinx()
ax2.plot(comparison_df['period'], comparison_df['theory swap'], label='theory swap', marker='o', color='orange')

# スワップポイントの軸の範囲を設定
ax1.set_ylim(min(comparison_df['average_buy_swap'].min(), comparison_df['average_sell_swap'].min()) * 1.1, max(comparison_df['average_buy_swap'].max(), comparison_df['average_sell_swap'].max()) * 1.1)
ax2.set_ylim(0, 2)  # 理論値の範囲を設定（適切に調整）

plt.title(f"Comparison of Swap Points for {pair}")
ax1.set_xlabel("Period")
ax1.set_ylabel("Swap Points")
ax2.set_ylabel("Theoretical Swap")

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')  # この行を追加

ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # 3ヶ月ごとに目盛りを設定
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()




# 割合グラフ
plt.subplot(2, 1, 2)
plt.plot(comparison_df['period'], comparison_df['buy_swap_ratio'], label='buy_swap_ratio', marker='o')
plt.plot(comparison_df['period'], comparison_df['sell_swap_ratio'], label='sell_swap_ratio', marker='o')

plt.title(f"comparison of swappoint ratio of {pair}")
plt.xlabel("period")
plt.ylabel("ratio")
plt.xticks(rotation=45, ha='right')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # 3ヶ月ごとに目盛りを設定
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.legend()
plt.tight_layout()

# 割合グラフを保存
plt.savefig(f"./png_dir/kabu_compare_bis_intrestrate_and_oandascraping_{pair}_{start_date}_{end_date}_{months_interval}_graph.png")

# グラフを表示
plt.show()
#