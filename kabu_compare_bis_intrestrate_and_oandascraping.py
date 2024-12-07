from kabu_oanda_swapscraping import scrape_from_oanda
from kabu_bis_intrestrate import filter_country_data
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import pytz

# Pandasの表示設定
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 100)


class Compare_Swap:
    CURRENCY_TIMEZONES = {
        'JPY': 'Asia/Tokyo',
        'ZAR': 'Africa/Johannesburg',
        'MXN': 'America/Mexico_City',
        'TRY': 'Europe/Istanbul',
        'CHF': 'Europe/Zurich',
        'NZD': 'Pacific/Auckland',
        'AUD': 'Australia/Sydney',
        'EUR': 'Europe/Berlin',
        'GBP': 'Europe/London',
        'USD': 'America/New_York',
        'CAD': 'America/Toronto',
        'NOK': 'Europe/Oslo',
        'SEK': 'Europe/Stockholm',
    }
    def __init__(self,pair,start_date,final_end,order_size,months_interval=1,window_size=30,cumulative_period=1,cumulative_unit="month"):
        directory = './csv_dir'
        rename_pair = pair.replace("/", "")
        
        try:
            start_date = start_date.strftime("%Y-%m-%d")
            final_end = final_end.strftime("%Y-%m-%d")
        except:
            pass
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
        self.window_size = window_size
        self.cumulative_period = cumulative_period
        self.cumulative_unit = cumulative_unit


        pair_splits = pair.split("/")
        interest_rates = {}
        for currency in pair_splits:
            interest_rate = self.download_interest_rate(currency)
            interest_rates[currency] = interest_rate

        self.interest_rates = interest_rates

    def change_utc_to_localtimezone(self,currency,date):
        try:
            date = date.strftime("%Y-%m-%d")
        except:
            pass
        utc = pytz.utc
        utc_datetime = utc.localize(datetime.strptime(date, "%Y-%m-%d"))
        local_timezone = pytz.timezone(self.CURRENCY_TIMEZONES.get(currency))
        local_datetime = utc_datetime.astimezone(local_timezone)
        return local_datetime


    def download_interest_rate(self, currency):
        country_name = self.currency_code_to_country_name(currency)
        directory = './csv_dir'

        start_date = self.change_utc_to_localtimezone(currency,self.start_date)
        final_end =  self.change_utc_to_localtimezone(currency,self.final_end)    
        target_start = start_date.date()
        target_end = final_end.date()


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
            'EUR': 'Euro_area',
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
        #pay attention to data type; we should change the data type of current_start and current_end to strftime.

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

        buy_values = [data['buy'] for date, data in filtered_data.items()]
        sell_values = [data['sell'] for date, data in filtered_data.items()]
        number_of_days = [data['number_of_days'] for date, data in filtered_data.items()]


        total_buy_swap = sum(buy_values)
        total_sell_swap = sum(sell_values)
        total_days = sum(number_of_days)

        average_buy_swap = total_buy_swap / total_days if total_days > 0 else 0
        average_sell_swap = total_sell_swap / total_days if total_days > 0 else 0


        return average_buy_swap, average_sell_swap


    def calculate_swap_cumulative_averages(self):
        start_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        current_end = start_date
        final_end = datetime.strptime(self.final_end, "%Y-%m-%d")
        results = []

        while current_end < final_end:
            if self.cumulative_unit == "month":
                current_end += relativedelta(months=self.cumulative_period) + relativedelta(days=-1)
            if self.cumulative_unit == "day":
                current_end += relativedelta(days=self.cumulative_period)


            filtered_data = self.get_data_range(self.swap_data,start_date.strftime("%Y-%m-%d"),current_end.strftime("%Y-%m-%d"))
            buy_values = [data['buy'] for date, data in filtered_data.items()]
            sell_values = [data['sell'] for date, data in filtered_data.items()]

            # 累積平均
            print(f"executing cumulative averages from {start_date} to {current_end}...")
            cumulative_avg_buy = pd.Series(buy_values).mean()
            cumulative_avg_sell = pd.Series(sell_values).mean()

            print(f"executing theorys from {start_date} to {current_end}...")
            theory = theory = sum(self.calculate_theory(date,start_date,current_end) for date in (start_date + timedelta(days=i) for i in range((current_end - start_date).days)))/(current_end-start_date).days

            results.append({
            "period": current_end,  # ここをdatetimeオブジェクトに変更
            "cumulative_avg_buy":cumulative_avg_buy,
            "cumulative_avg_sell":cumulative_avg_sell,
            "theory":theory
            })

        results = pd.DataFrame(results)

        return results


    def calculate_swap_moving_averages(self):
        start_date = self.start_date
        final_end = self.final_end
    
        filtered_data = self.get_data_range(self.swap_data,start_date,final_end)
        buy_values = [data['buy'] for date, data in filtered_data.items()]
        sell_values = [data['sell'] for date, data in filtered_data.items()]

        # スライディングウィンドウ
        moving_avg_buy = pd.Series(buy_values).rolling(window=self.window_size).mean()
        moving_avg_sell = pd.Series(sell_values).rolling(window=self.window_size).mean()

        moving_avg_buy = moving_avg_buy.dropna()
        moving_avg_sell = moving_avg_sell.dropna()


        results = []
        print(len(moving_avg_buy)-1)
        for i in range(len(moving_avg_buy)-1):
            first_key = list(filtered_data.keys())[i]
            last_key = list(filtered_data.keys())[i+self.window_size]

            #print(f"first_key:{first_key}")
            #print(f"last_key:{last_key}")

            theory = theory = sum(self.calculate_theory(date,first_key,last_key) for date in (first_key + timedelta(days=i) for i in range((last_key - first_key).days)))/(last_key-first_key).days

            

            results.append({
                "period":datetime.strptime(first_key,"%Y-%m-%d"),  # ここをdatetimeオブジェクトに変更
                "moving_avg_buy":moving_avg_buy[i+self.window_size-1],
                "moving_avg_sell":moving_avg_sell[i+self.window_size-1],
                "theory": theory
                })

        results = pd.DataFrame(results)

        return results


    def get_interest_rate_list(self,current_start, current_end):
        pair_splits = self.pair.split("/")
        interest_list = []
        for currency in pair_splits:
            current_start = self.change_utc_to_localtimezone(currency,current_start)
            current_end = self.change_utc_to_localtimezone(currency,current_end)
            current_start = current_start.strftime("%Y-%m-%d")
            current_end = current_end.strftime("%Y-%m-%d")

            interest_rate = self.interest_rates[currency]
            #print(interest_rate)
            #exit()
            interest_rate['TIME_PERIOD:Time period or range'] = pd.to_datetime(interest_rate['TIME_PERIOD:Time period or range'])



            filtered_interest_rate = interest_rate[(current_start <= interest_rate['TIME_PERIOD:Time period or range']) & (interest_rate['TIME_PERIOD:Time period or range'] <= current_end)]

            # DataFrameに変換
            df = pd.DataFrame(filtered_interest_rate)

            # 必要な列を取り出す
            result = df[["TIME_PERIOD:Time period or range", "OBS_VALUE:Observation Value"]]
            result_dict = result.set_index('TIME_PERIOD:Time period or range')['OBS_VALUE:Observation Value'].to_dict()

            interest_list.append(result_dict)
        return interest_list

    def calculate_theory(self,date,current_start,current_end):
        interest_list = self.get_interest_rate_list(current_start,current_end)
        pair_splits = self.pair.split("/")
        date_0 = self.change_utc_to_localtimezone(pair_splits[0],date)
        date_1 = self.change_utc_to_localtimezone(pair_splits[1],date)
        date_0 = date_0.date()
        date_1 = date_1.date()
        date_0 = pd.Timestamp(date_0)
        date_1 = pd.Timestamp(date_1)

        theory = (interest_list[0].get(date_0) - interest_list[1].get(date_1)) * self.order_size / 100 * 1 / 365


        return theory

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

            theory = sum(self.calculate_theory(date,current_start,current_end) for date in (current_start + timedelta(days=i) for i in range((current_end - current_start).days)))/(current_end-current_start).days

            results.append({
                "period": current_start,  # ここをdatetimeオブジェクトに変更
                "average_buy_swap": avg_buy,
                "average_sell_swap": avg_sell,
                "theory": theory
            })
            current_start = current_end + relativedelta(days=1)

        comparison_df = pd.DataFrame(results)
        return comparison_df 


# グラフを作成
def makegraph(arg1, arg2, results,graphname):
    plt.figure(figsize=(10, 12))

    # スワップポイントグラフ
    plt.subplot(2, 1, 1)
    ax1 = plt.gca()  # 現在のAxesを取得
    ax1.plot(results['period'], results[arg1], label=arg1, marker='o')
    ax1.plot(results['period'], results[arg2], label=arg2, marker='o')

    # 理論値をプロットするための新しいy軸を作成
    ax2 = ax1.twinx()
    ax2.plot(results['period'], results['theory'], label='theory', marker='o', color='orange')

    # スワップポイントの軸の範囲を設定
    ax1.set_ylim(min(results[arg1].min(), results[arg2].min()) * 1.1, max(results[arg1].max(), results[arg2].max()) * 1.1)
    ax2.set_ylim(0, 2)  # 理論値の範囲を設定（適切に調整）

    plt.title(f"Comparison of Swap Points for {pair} by {graphname}")
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
    plt.plot(results['period'], results[arg1]/results['theory'], label='buy_swap_ratio', marker='o')
    plt.plot(results['period'], results[arg2]/results['theory'], label='sell_swap_ratio', marker='o')

    plt.title(f"comparison of swappoint ratio of {pair} by {graphname}")
    plt.xlabel("period")
    plt.ylabel("ratio")
    plt.xticks(rotation=45, ha='right')
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # 3ヶ月ごとに目盛りを設定
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.legend()
    plt.tight_layout()

    # 割合グラフを保存
    plt.savefig(f"./png_dir/kabu_compare_bis_intrestrate_and_oandascraping_{pair}_{start_date}_{end_date}_{months_interval}_{graphname}.png")

    # グラフを表示
    plt.show()


if __name__ == "__main__":
    # 使用例
    pair = "USD/JPY"
    start_date = "2019-04-01"
    end_date = "2024-10-31"
    order_size = 10000 if pair != "ZAR/JPY" and pair != "HKD/JPY" else 100000
    months_interval = 1

    scrapeswap = Compare_Swap(pair,start_date,end_date, order_size, months_interval)

    comparison_df = scrapeswap.multiple_period_swap_comparison()
    print(comparison_df)
    cumulative_averages = scrapeswap.calculate_swap_cumulative_averages()
    moving_averages = scrapeswap.calculate_swap_moving_averages()

    pair = pair.replace("/","")
    # 結果をCSVファイルとして保存
    comparison_df.to_csv(f"./csv_dir/kabu_compare_bis_intrestrate_and_oandascraping_{pair}_{start_date}_{end_date}_{months_interval}_averages.csv", index=False, encoding='utf-8-sig')
    #print(min(comparison_df['average_buy_swap'].min(), comparison_df['average_sell_swap'].min()))
    #print(max(comparison_df['average_buy_swap'].max(), comparison_df['average_sell_swap'].max()))
    #exit()
    #
    moving_averages.to_csv(f"./csv_dir/kabu_compare_bis_intrestrate_and_oandascraping_{pair}_{start_date}_{end_date}_{months_interval}_moving_avg.csv", index=False, encoding='utf-8-sig')
    cumulative_averages.to_csv(f"./csv_dir/kabu_compare_bis_intrestrate_and_oandascraping_{pair}_{start_date}_{end_date}_{months_interval}_cumulative_avg.csv", index=False, encoding='utf-8-sig')
    makegraph("average_buy_swap","average_sell_swap",comparison_df,"average")
    makegraph("moving_avg_buy","moving_avg_sell",moving_averages,"moving_avg")
    makegraph("cumulative_avg_buy","cumulative_avg_sell",cumulative_averages,"cumulative_avg")