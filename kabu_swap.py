from datetime import datetime, timedelta, time
import time as time_module
import holidays
import requests
from bs4 import BeautifulSoup
import pytz
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager
import os
import pandas as pd
from kabu_library import fetch_currency_data

class SwapCalculator:
    NY_CLOSE_TIME = time(17, 0)  # NYクローズの時刻は午後5時（夏時間・冬時間は自動調整）
    NY_TIMEZONE = pytz.timezone("America/New_York")  # ニューヨーク時間帯
    original_timezone = pytz.utc
    # 通貨ごとのタイムゾーンを定義
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
    def __init__(self, website, pair, start_date, end_date, interval="1d",link=None):
        self.timezones = self.get_timezones_from_pair(pair)
        self.each_holidays = self.get_holidays_from_pair(pair)  # 祝日データをインスタンスに保持
        self.holiday_cache = {currency: {} for currency in self.each_holidays.keys()}  # 通貨ごとの祝日判定結果のキャッシュ
        self.website = website

        if website == "minkabu":
            self.per_order_size = 10000 if pair in ['ZARJPY=X', 'MXNJPY=X'] else 1000 #MINKABU
            self.fetcher = ScrapeFromMinkabu()
            if not "JPY" in pair:
                modified_pair = self.arrange_pair_format(pair)[1] + "JPY=X"
                self.modified_pair_data = fetch_currency_data(modified_pair,start_date, end_date, interval,link=link)
        elif website == "oanda":
            self.per_order_size = 100000 if pair in ['ZARJPY=X','HKDJPY=X'] else 10000
            self.fetcher = ScrapeFromOanda(pair,start_date,end_date)
        else:
            raise ValueError("Invalid website specified")
        
        self.swap_points_dict = self.fetcher.swap_points_dict


    def arrange_pair_format(self,pair):
        if "=X" in pair:
            currencies = pair.replace("=X","")
        if "_" in pair:
            currencies = currencies.split("_")
        if "/" in pair:
            currencies = pair.split('/')
        if not "_" in pair and not "/" in pair:
            currencies = [currencies[:3],currencies[3:]]
        return currencies


    def get_timezones_from_pair(self, pair):
        currencies = self.arrange_pair_format(pair)
        return [self.CURRENCY_TIMEZONES.get(currency) for currency in currencies]


    def get_holidays_from_pair(self, pair):
        currencies = self.arrange_pair_format(pair)
        
        # 例外処理（ユーロなど）
        exceptions = {
            "EUR": "DE",   # ユーロはドイツを代表例に設定
            "XAU": None,   # 金は特定の国の祝日は不要
            "XAG": None    # 銀も同様
        }

        # 祝日を保持する辞書
        holidays_dict = {}

        # ベース通貨とクオート通貨それぞれの祝日を追加
        for currency in currencies:
            # 例外処理をチェック
            country_code = exceptions.get(currency, currency[:2])
            if country_code:  # Noneの場合はスキップ
                holidays_dict[currency] = holidays.CountryHoliday(country_code)

        return holidays_dict

    # 祝日をチェックするメソッド
    def is_holiday(self, date, currency):
        # 入力が時刻を含むかどうかを判断
        for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d'):
            try:
                # 試行して日付を解析
                date = datetime.strptime(str(date), fmt).date()
                break  # 成功したらループを抜ける
            except ValueError:
                continue  # 失敗したら次の形式を試す


        if date not in self.holiday_cache[currency]:  # キャッシュに結果がない場合のみ計算
            self.holiday_cache[currency][date] = date in self.each_holidays[currency]
        return self.holiday_cache[currency][date]

    # 2営業日後の日付を計算するメソッド
    def add_business_days(self, start_date, num_units, trading_days_set, interval, pair, swap_flag=True):
        # 現在の日時をニューヨーク時間に変換
        #start_date = start_date.astimezone(self.NY_TIMEZONE)

        current_date = start_date
        added_units = 0


        while added_units < num_units:
            # timeframe_minutesを日数に変換

            current_date_before = current_date

            if interval == "1d":  # 日足の場合
                current_date += timedelta(days=1)
            elif interval == "H1":
                current_date += timedelta(hours=1)
            elif interval == "M1":
                current_date += timedelta(minutes=1)
            

            # 土日でなく、かつ祝日でない、もしくはtrading_daysに含まれている場合
            if swap_flag:
                if (current_date.weekday() < 5 and 
                    (not self.is_holiday(current_date.astimezone(pytz.timezone(self.timezones[0])),self.arrange_pair_format(pair)[0]) and 
                     not self.is_holiday(current_date.astimezone(pytz.timezone(self.timezones[1])),self.arrange_pair_format(pair)[1]))):
                    # NYクローズを跨いでいるかを判定
                    if not self.crossed_ny_close(current_date_before.astimezone(self.NY_TIMEZONE)) and self.crossed_ny_close(current_date.astimezone(self.NY_TIMEZONE)) or interval == "1d":
                        added_units += 1
            else:
                if (current_date.weekday() < 5 and 
                    (not self.is_holiday(current_date.astimezone(pytz.timezone(self.timezones[0])),self.arrange_pair_format(pair)[0]) and 
                     not self.is_holiday(current_date.astimezone(pytz.timezone(self.timezones[1]),self.arrange_pair_format(pair)[1]))) 
                     or current_date in trading_days_set):
                    # NYクローズを跨いでいるかを判定
                    if not self.crossed_ny_close(current_date_before.astimezone(self.NY_TIMEZONE)) and self.crossed_ny_close(current_date.astimezone(self.NY_TIMEZONE)) or interval == "1d":
                        added_units += 1

        return current_date.astimezone(self.original_timezone)

    # NYクローズを跨いだかを判定するメソッド
    def crossed_ny_close(self, dt):

        # 夏時間・冬時間を考慮しつつ午後5時を跨いでいればTrueを返す
        if dt.time() >= self.NY_CLOSE_TIME:
            return True
        return False


    # ロールオーバーの日数を計算するメソッド
    def calculate_rollover_days(self, open_date, current_date, trading_days_set, pair):
        try:
            rollover_days = (self.add_business_days(current_date, 2, trading_days_set, "1d", pair) - self.add_business_days(open_date, 2, trading_days_set, "1d", pair)).days


            open_date = open_date.astimezone(self.NY_TIMEZONE)
            current_date = current_date.astimezone(self.NY_TIMEZONE)
            open_time = open_date.time()
            current_time = current_date.time()

            if open_time <= self.NY_CLOSE_TIME <= current_time:
                if open_time== self.NY_CLOSE_TIME == current_time:
                    pass
                elif open_time == self.NY_CLOSE_TIME and self.NY_CLOSE_TIME != current_time:
                    pass
                elif open_time != self.NY_CLOSE_TIME and self.NY_CLOSE_TIME == current_time:
                    rollover_days += 1
                    print("added 1 to rollover_days in 'if open_time != self.NY_CLOSE_TIME and self.NY_CLOSE_TIME == current_time' in 'if open_time <= self.NY_CLOSE_TIME <= current_time' ")
                elif open_time != self.NY_CLOSE_TIME and self.NY_CLOSE_TIME != current_time:
                    rollover_days += 1
                    print("added 1 to rollover_days in 'if open_time != self.NY_CLOSE_TIME and self.NY_CLOSE_TIME != current_time' in 'if open_time <= self.NY_CLOSE_TIME <= current_time' ")

            elif current_time <= self.NY_CLOSE_TIME <= open_time:
                if current_time == self.NY_CLOSE_TIME == open_time:
                    pass
                elif current_time != self.NY_CLOSE_TIME and self.NY_CLOSE_TIME == open_time:    #open_time=17:00 current_time=16:59
                    pass
                elif current_time == self.NY_CLOSE_TIME and self.NY_CLOSE_TIME != open_time:    #open_time=17:01 current_time=17:00
                    rollover_days += 1
                    print("added 1 to rollover_days in 'if current_time == self.NY_CLOSE_TIME and self.NY_CLOSE_TIME != open_time' in 'elif current_time <= self.NY_CLOSE_TIME <= open_time' ")
                elif current_time != self.NY_CLOSE_TIME and self.NY_CLOSE_TIME != open_time:    #open_time=17:02 current_time=16:59
                    pass

            elif  current_time <= open_time <= self.NY_CLOSE_TIME:
                if current_time == open_time == self.NY_CLOSE_TIME:
                    pass
                elif current_time != open_time and open_time == self.NY_CLOSE_TIME: #open_time=17:00 current_time=16:59
                    pass
                elif current_time == open_time and open_time != self.NY_CLOSE_TIME: #open_time=16:59 current_time=16:59
                    pass
                elif current_time != open_time and open_time != self.NY_CLOSE_TIME: #open_time=16:59 current_time=16:58
                    rollover_days += 1
                    print("added 1 to rollover_days in 'elif current_time != open_time and open_time != self.NY_CLOSE_TIME' in 'elif current_time <= open_time <= self.NY_CLOSE_TIME' ")

            elif open_time <= current_time <= self.NY_CLOSE_TIME:
                if open_time == current_time == self.NY_CLOSE_TIME:
                    pass
                elif open_time != current_time and current_time == self.NY_CLOSE_TIME:  #open_time=16:59 current_time=17:00
                    rollover_days += 1
                    print("added 1 to rollover_days in 'if open_time != current_time and current_time == self.NY_CLOSE_TIME' in 'elif open_time <= current_time <= self.NY_CLOSE_TIME' ")
                elif open_time == current_time and current_time != self.NY_CLOSE_TIME:  #open_time=16:59 current_time=16:59
                    pass
                elif open_time != current_time and current_time != self.NY_CLOSE_TIME:  #open_time=16:58 current_time=16:59
                    pass
                

            elif self.NY_CLOSE_TIME <= open_time <= current_time:
                if self.NY_CLOSE_TIME == open_time == current_time:
                    pass
                elif self.NY_CLOSE_TIME != open_time and open_time == current_time: #open_time=17:01 current_time=17:01
                    pass
                elif self.NY_CLOSE_TIME == open_time and open_time != current_time: #open_time=17:00 current_time=17:01
                    pass
                elif self.NY_CLOSE_TIME != open_time and open_time != current_time: #open_time=17:01 current_time=17:02
                    pass


            elif self.NY_CLOSE_TIME <= current_time <= open_time:
                if self.NY_CLOSE_TIME == current_time == open_time:
                    pass
                elif self.NY_CLOSE_TIME != current_time and current_time == open_time:  #open_time=17:01 current_time=17:01
                    pass
                elif self.NY_CLOSE_TIME == current_time and current_time != open_time:  #open_time=17:01 current_time=17:00
                    rollover_days += 1
                    print("added 1 to rollover_days in 'elif self.NY_CLOSE_TIME == current_time and current_time != open_time' in 'elif self.NY_CLOSE_TIME <= current_time <= open_time' ")
                elif self.NY_CLOSE_TIME != current_time and current_time != open_time:  #open_time=17:02 current_time=17:01
                    rollover_days += 1
                    print("added 1 to rollover_days in 'self.NY_CLOSE_TIME != current_time and current_time != open_time' in 'elif self.NY_CLOSE_TIME <= current_time <= open_time' ")

            
            
        except IndexError as e:
            print(f"Error in calculating rollover days: {e}")
            return 0
        return rollover_days

    def get_total_swap_points(self, pair, position, open_date, current_date, order_size, trading_days):
        trading_days_set = set(trading_days)
        rollover_days = self.calculate_rollover_days(open_date, current_date, trading_days_set, pair)
        

        if self.website == "minkabu":
            if pair not in self.swap_points_dict:
                return 0.0
            swap_value = self.fetcher.swap_points_dict[pair].get('buy' if "Buy" in position else 'sell', 0)
            if not "JPY" in pair:
                end_date = open_date + timedelta(days=rollover_days)
                return sum(swap_value / self.modified_pair_data[open_date:end_date] * order_size / self.per_order_size)
            else:
                return swap_value * rollover_days * order_size / self.per_order_size

        if self.website == "oanda":
            data = self.swap_points_dict
            # swap_value の初期化
            swap_value = 0

            # open_date から current_date までの日付をループ
            current = open_date
            while current <= current_date:
                date_str = current.strftime("%Y-%m-%d")  # 文字列に変換
                swap_value += data.get(date_str, {}).get('buy' if "Buy" in position else 'sell', 0)
                current += timedelta(days=1)  # 次の日に進める

            return swap_value * order_size / self.per_order_size



class ScrapeFromMinkabu:
    def __init__(self):
        url = 'https://fx.minkabu.jp/hikaku/moneysquare/spreadswap.html'
        self.html = self.get_html(url)
        self.swap_points = self.parse_swap_points(self.html)
        self.swap_points = self.rename_swap_points(self.swap_points)
        self.swap_points_dict = self._create_swap_dict(self.swap_points)

    def _create_swap_dict(self, swap_points):
        swap_dict = {}
        for point in swap_points:
            pair = point.get('通貨名')
            buy_swap = self._safe_float(point.get('買スワップ', 0))
            sell_swap = self._safe_float(point.get('売スワップ', 0))
            swap_dict[pair] = {'buy': buy_swap, 'sell': sell_swap}
        return swap_dict
    
    def _safe_float(self, value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    # URLからHTMLデータを取得する関数
    def get_html(self,url):
        response = requests.get(url)
        response.raise_for_status()  # エラーが発生した場合、例外を発生させる
        return response.text

    # 不換空白文字を削除する関数
    def clean_text(self,text):
        return text.replace('\xa0', '').replace('円', '').strip()

    # HTMLを解析してスワップポイントを抽出する関数
    def parse_swap_points(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        swap_table = soup.find('table')  # スワップポイントのテーブルを特定
        if not swap_table:
            raise ValueError("スワップポイントのテーブルが見つかりません")

        rows = swap_table.find_all('tr')

        #headers = [clean_text(th.get_text(strip=True)) for th in rows[0].find_all('th')]
        # 'clean_text'の処理をその場で行う
        headers = [th.get_text(strip=True).replace('\xa0', '').replace('円', '').strip() for th in rows[0].find_all('th')]


        swap_points = []
        for row in rows[1:]:
            cells = row.find_all(['th', 'td'])
            if len(cells) != len(headers):
                continue
            swap_point = {headers[i]: self.clean_text(cells[i].get_text(strip=True)) for i in range(len(headers))}
            swap_points.append(swap_point)

        return swap_points

    def rename_swap_points(self, swap_points):
        for item in swap_points:
            item['通貨名'] = self.convert_currency_name(item['通貨名'])
        return swap_points

    # 通貨名を変換する関数
    def convert_currency_name(self,currency_name):
        currency_mappings = {
            '米ドル/カナダドル': 'USDCAD=X',
            'ユーロ/米ドル': 'EURUSD=X',
            '英ポンド/米ドル': 'GBPUSD=X',
            '豪ドル/米ドル': 'AUDUSD=X',
            'NZドル/米ドル': 'NZDUSD=X',
            'ユーロ/英ポンド': 'EURGBP=X',
            '豪ドル/NZドル': 'AUDNZD=X',
            '米ドル/': 'USDJPY=X',
            'ユーロ/': 'EURJPY=X',
            '英ポンド/': 'GBPJPY=X',
            '豪ドル/': 'AUDJPY=X',
            'NZドル/': 'NZDJPY=X',
            'カナダドル/': 'CADJPY=X',
            '南アフリカランド/': 'ZARJPY=X',
            'トルコリラ/': 'TRYJPY=X',
            'メキシコペソ/': 'MXNJPY=X'
        }
        for key, value in currency_mappings.items():
            if key in currency_name:
                return value
        return currency_name
    

class ScrapeFromOanda:
    def __init__(self,pair, start_date, end_date):
        
        directory = './csv_dir'
        rename_pair = pair.replace("/", "")
        rename_pair = rename_pair.replace("=X","")
        self.rename_pair = rename_pair
        

        try:
            target_start = datetime.strptime(start_date, '%Y-%m-%d')
            target_end = datetime.strptime(end_date, '%Y-%m-%d')
        except:
            target_start = start_date
            target_end = end_date
        
        
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
            swap_data_t = swap_data.T
            self.swap_points_dict = swap_data_t.to_dict()
        else:
            print(f"scrape_from_oanda({pair}, {start_date}, {end_date})")
            self.swap_points_dict = self.scrape_from_oanda(pair, start_date, end_date)


    def arrange_pair_format(self,pair):
        if "=X" in pair:
            pair = pair.replace("=X","")
        if "_" in pair:
            pair = pair.replace("_", "/")
        if "/" in pair:
            pass
        if not "_" in pair and not "/" in pair:
            pair = pair[:3] + "/" + pair[3:]
        return pair
    
    def scrape_from_oanda(self, pair, start_date, end_date):

        pair = self.arrange_pair_format(pair)
        
        # 日付をdatetime形式に変換
        # datetimeオブジェクトであるかチェック
        if isinstance(start_date, datetime) and isinstance(end_date, datetime):
            start_date = start_date.date().strftime("%Y-%m-%d")
            start_date = datetime.strptime(start_date,"%Y-%m-%d")
            end_date = end_date.date().strftime("%Y-%m-%d")
            end_date = datetime.strptime(end_date,"%Y-%m-%d")

        if isinstance(start_date, str) and isinstance(end_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
            end_date = datetime.strptime(end_date, "%Y-%m-%d")


        # データ取得の制限を確認
        if start_date < datetime(2019, 4, 1):
            print("2019年4月以前のデータはありません。")
            start_date = datetime(2019, 4, 1)

        # データを保存するための辞書
        all_data = {}

        # Chromeのオプションを設定
        options = Options()
        options.add_argument('--headless')  # ヘッドレスモード（ブラウザを表示せずに実行）

        # WebDriverを初期化
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)

        # スワップポイントの情報が掲載されているURL
        url = "https://www.oanda.jp/course/ny4/swap"
        driver.get(url)

        # 通貨ペアを選択
        currency_pair = Select(driver.find_element(By.CLASS_NAME, 'st-Select'))
        currency_pair.select_by_visible_text(pair)  # 引数で指定された通貨ペアを選択

        # 指定された日付範囲で日毎のデータを取得
        current_date = start_date
        while current_date <= end_date:
            # 年月のテキストを作成
            year_month = current_date.strftime("%Y年%m月")

            # 月の選択
            month_select = Select(driver.find_elements(By.CLASS_NAME, 'st-Select')[1])
            try:
                month_select.select_by_visible_text(year_month)  # 指定された年月を選択
            except:
                print(f"{year_month} のデータは存在しません。")
                current_date += timedelta(days=1)
                continue
            
            # データ取得のために少し待機
            time_module.sleep(2)

            # テーブルを取得
            try:
                swap_table = driver.find_element(By.CLASS_NAME, 'tr-SwapHistory_Table')
            except:
                print(f"{year_month} のテーブルが見つかりません。")
                current_date += timedelta(days=1)
                continue

            # テーブルの行を取得し、該当する日付の行からデータを取得
            rows = swap_table.find_elements(By.TAG_NAME, 'tr')
            for row in rows[1:]:  # 最初の行はヘッダーなのでスキップ
                # 日付の列とデータの列を取得
                date_col = row.find_element(By.TAG_NAME, 'th')
                data_cols = row.find_elements(By.TAG_NAME, 'td')


                # 日付がcurrent_dateと一致するか確認
                date_text = re.sub(r'（.*?）', '', date_col.text)  # 曜日を除去

                # 各項目を辞書に保存
                if len(data_cols) == 3:  # 期待するデータ数を確認
                    sell_text = data_cols[0].text.strip()
                    buy_text = data_cols[1].text.strip()
                    days_text = data_cols[2].text.strip()

                    # 日付から「月」と「日」を除外し、日付部分を整数として取得
                    match = re.match(r'(\d{2})月(\d{2})日', date_text)
                    if match:
                        day = int(match.group(2))  # 「日」の部分を整数に変換
                        date_str = f"{current_date.year}-{current_date.month:02}-{day:02}"  # 年月日形式に変換
                    else:
                        print("日付形式が不正です:", date_str)
                        continue
                    
                    all_data[date_str] = {
                        'sell': float(sell_text) if sell_text else 0,
                        'buy': float(buy_text) if buy_text else 0,
                        'number_of_days': int(days_text) if days_text else 0
                    }

            # 次の月へ進む
            current_date += timedelta(days=31)
            current_date = current_date.replace(day=1)  # 次の月の1日に設定

        # WebDriverを終了
        driver.quit()


        # スクレイピングで得た最初と最後の日付を取得
        dates = sorted(all_data.keys())
        actual_start_date = dates[0] if dates else start_date
        actual_end_date = dates[-1] if dates else end_date
        pd.DataFrame(all_data).to_csv(f'./csv_dir/kabu_oanda_swapscraping_{self.rename_pair}_from{actual_start_date}_to{actual_end_date}.csv', index=False, encoding='utf-8')
        print(f"saved ./csv_dir/kabu_oanda_swapscraping_{self.rename_pair}_from{actual_start_date}_to{actual_end_date}.csv")

        # 取得したデータを返す
        return all_data

        
    



if __name__ == "__main__":
    order_size = 1000

    start_date = datetime(2024, 9, 16, 6, 1)
    end_date = datetime(2024, 9, 17, 6, 1)

    calculator = SwapCalculator("oanda", 'USDJPY=X', start_date, end_date,interval="M1")

    total_swap_points = calculator.get_total_swap_points('USDJPY=X', "Buy", start_date, end_date, order_size, [])
    print(total_swap_points)

    #a = calculator.get_total_swap_points('USDJPY=X', "Buy", datetime(2024, 6, 4), datetime(2024, 6, 10), order_size, [])
    #b = calculator.get_total_swap_points('USDJPY=X', "Buy", datetime(2024, 6, 11), datetime(2024, 6, 11), order_size, [])
    #c = calculator.get_total_swap_points('USDJPY=X', "Buy", datetime(2024, 6, 12), datetime(2024, 6, 12), order_size, [])
    #print(f"a, b, a+b+c: {a}, {b}, {c}, {a + b + c}")
