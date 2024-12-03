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


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time_module.time()
        result = func(*args, **kwargs)
        elapsed_time = time_module.time() - start_time
        print(f"Method {func.__name__}: {elapsed_time:.6f} seconds")
        return result
    return wrapper


class SwapCalculator:
    NY_CLOSE_TIME = time(17, 0)  # NYクローズの時刻は午後5時（夏時間・冬時間は自動調整）
    #NY_CLOSE_TIME = time(7, 0)  # NYクローズの時刻は午後5時で、それを日本時間に直すと14時間後の午前7時、oanda証券のスワップカレンダーが日本時間を基準としているので日本時間で計算すべし
    #NY_TIMEZONE = pytz.timezone("America/New_York")  # ニューヨーク時間帯
    JP_TIMEZONE = pytz.timezone("Asia/Tokyo")
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
    def __init__(self, website, pair, start_date, end_date, interval="1d"):
        self.timezones = self.get_timezones_from_pair(pair)
        self.website = website

        if website == "minkabu":
            self.per_order_size = 10000 if pair in ['ZARJPY=X', 'MXNJPY=X'] else 1000 #MINKABU
            self.fetcher = ScrapeFromMinkabu()
        elif website == "oanda":
            self.per_order_size = 100000 if pair in ['ZARJPY=X','HKDJPY=X'] else 10000
            self.fetcher = ScrapeFromOanda(pair,start_date,end_date)
        else:
            raise ValueError("Invalid website specified")
        
        self.swap_points_dict = self.fetcher.swap_points_dict

        self.business_days = self.generate_business_days(pair, start_date, end_date)

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

    def get_holidays_from_pair(self, pair,start_date,end_date):
        currencies = self.arrange_pair_format(pair)
        
        # 例外処理（ユーロなど）
        exceptions = {
            "EUR": "DE",   # ユーロはドイツを代表例に設定
            "XAU": None,   # 金は特定の国の祝日は不要
            "XAG": None    # 銀も同様
        }

        # 祝日を保持する辞書
        holidays_dict = {}
        start_date = start_date -timedelta(days=1)
        end_date = end_date + timedelta(days=1)
        years = list(range(start_date.year - 1, end_date.year + 1))


        # ベース通貨とクオート通貨それぞれの祝日を追加
        i = 0
        for currency in currencies:
            # 例外処理をチェック
            country_code = exceptions.get(currency, currency[:2])
            print(country_code)
            if country_code:  # Noneの場合はスキップ
                start_date = start_date.astimezone(pytz.timezone(self.timezones[i]))
                end_date = end_date.astimezone(pytz.timezone(self.timezones[i]))
                start_date_date = start_date.date()
                end_date_date = end_date.date()
                holidays_dict[currency] = sorted([
                d for d in holidays.CountryHoliday(country_code,years=years).keys() if start_date_date <= d <= end_date_date
            ])
            i += 1


        return holidays_dict


    def is_ny_business_day(self, date):
        date = date.astimezone(pytz.timezone('America/New_York'))
        """
        ニューヨーク時間で、日曜日17:00〜金曜日16:59の間（日本時間では月曜日7:00~土曜日6:59）であればTrue、それ以外はFalseを返す関数。
        """

        # 月曜日7:00以降、または月〜金曜日、土曜日7:00前は営業日
        if date.weekday() == 6 and date.time() < self.NY_CLOSE_TIME: # 日曜日の17:00前
            return False
        elif date.weekday() == 6 and date.time() >= self.NY_CLOSE_TIME:
            return True
        elif date.weekday() in [0, 1, 2, 3]:  # 月〜木曜日
            return True
        elif date.weekday() == 4 and date.time() < self.NY_CLOSE_TIME:  # 金曜日17:00前
            return True

        # その他（営業日外）
        return False

    def convert_ny_close_to_local(self, holiday_date, timezone_str):
        """ニューヨーク時間17:00を指定されたタイムゾーンに変換"""
        local_timezone = pytz.timezone(timezone_str)
        holiday_date = datetime.combine(holiday_date, time(0, 0))
        holiday_date = local_timezone.localize(holiday_date)

        # NYクローズ時間をNewyork時間で設定(NYクローズはサマータイムの有無に関わらずnewyork時間では17:00で変わらないからこれを基準とする)
        ny_timezone = pytz.timezone("America/New_York")
        ny_start = holiday_date.astimezone(ny_timezone)
        ny_start = ny_start.replace(hour=17, minute=0, second=0)

        # 現地タイムゾーンに変換
        local_start = ny_start.astimezone(local_timezone)

        local_end = local_start + timedelta(hours=23, minutes=59)

        return local_start, local_end

    def get_holiday_time_ranges(self, pair, start_date, end_date):
        """祝日を特定し、各通貨の現地時間でholiday_time_rangesを生成"""
        holidays_dict = self.get_holidays_from_pair(pair, start_date, end_date)

        holiday_time_ranges = {}
        # holidays_dictのキー（通貨）があっても、空のリストがある場合には空のリストをセット
        currencies = self.arrange_pair_format(pair)
        for currency in currencies:
            if currency not in holidays_dict or not holidays_dict[currency]:
                holiday_time_ranges[currency] = []
            else:
                # 祝日がある場合には、通常の処理
                timezone = self.CURRENCY_TIMEZONES.get(currency)
                for holiday_date in holidays_dict[currency]:
                    start_time, end_time = self.convert_ny_close_to_local(holiday_date, timezone)
                    holiday_time_ranges[currency] = holiday_time_ranges.get(currency, []) + [(start_time, end_time)]
        return holiday_time_ranges

    # 2営業日後の日付を計算するメソッド
    def generate_business_days(self, pair, start_date, end_date):
        # 通貨ペアの祝日を取得
        holiday_time_ranges = self.get_holiday_time_ranges(pair, start_date, end_date)
        
        # 通貨ペアの2国分の祝日時間を分ける
        currencies = self.arrange_pair_format(pair)
    
        # 全体の日時範囲を作成（start_date から end_date まで）
        start_datetime = datetime.combine(start_date-timedelta(days=1), datetime.min.time())   #2営業日後を計算で使用する都合上、、start_date,end_dateを範囲としてしまうと、それを過ぎた場合の営業日が計算できなくなるので時差を考慮してstart_dateは1日前まで、end_dateは１ヶ月余裕を持って範囲指定する
        end_datetime = datetime.combine(end_date+timedelta(days=30), datetime.max.time())
    
        # 全体の日時リスト（1分単位で全ての日付）
        all_dates = pd.date_range(start=start_datetime, end=end_datetime, freq='min', tz=pytz.UTC).to_pydatetime().tolist()
    
        # 祝日と休日の日時を除外して辞書に格納
        business_days_dict = {}
        for date in all_dates:
            # 祝日や休日の時間範囲に含まれていない場合
            if not any(holiday_start <= date.astimezone(pytz.timezone(self.timezones[0])) <= holiday_end for holiday_start, holiday_end in holiday_time_ranges[currencies[0]]):
                if not any(holiday_start <= date.astimezone(pytz.timezone(self.timezones[1])) <= holiday_end for holiday_start, holiday_end in holiday_time_ranges[currencies[1]]):
                
                    # ニューヨーク時間基準で営業日かつ祝日でない場合
                    if self.is_ny_business_day(date):
                        business_days_dict[date] = True
    
        return business_days_dict


    def add_business_days(self, start_datetime, num_units, trading_days_set, interval="1d", swap_flag=True):
        """
        任意の営業日単位で日付を進める
        :param start_datetime: 開始日時
        :param add_unit: 進めたい営業日数
        :param interval: 進める単位（例："1d"は1日, "1h"は1時間）
        :return: 進めた営業日
        """
        try:
            start_datetime = start_datetime.astimezone(pytz.utc)
        except:
            pass

        if interval == "M1":
            interval = "1m"
        
        current_datetime = start_datetime
        added_units = 0

        while added_units < num_units:
            current_datetime += pd.Timedelta(interval)

            # 次の日付が営業日であればカウント
            if swap_flag and current_datetime in self.business_days:                
                added_units += 1
            elif not swap_flag and (current_datetime in self.business_days and trading_days_set in self.business_days):
                added_units += 1

        return current_datetime


    # ロールオーバーの日数を計算するメソッド
    def calculate_rollover_days(self, open_date, current_date, trading_days_set):
        try:
            rollover_days = (self.add_business_days(current_date, 2, trading_days_set, "1d") - self.add_business_days(open_date, 2, trading_days_set, "1d")).days


            open_date = open_date.astimezone(pytz.timezone("America/New_York"))
            current_date = current_date.astimezone(pytz.timezone("America/New_York"))
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
        rollover_days = self.calculate_rollover_days(open_date, current_date, trading_days_set)

        if rollover_days == 0:
            return 0
        
        elif rollover_days >= 1:
            if self.website == "minkabu":
                if pair not in self.swap_points_dict:
                    return 0.0
                swap_value = self.fetcher.swap_points_dict[pair].get('buy' if "Buy" in position else 'sell', 0)
                return swap_value * rollover_days * order_size / self.per_order_size

            if self.website == "oanda":
                data = self.swap_points_dict
                # swap_value の初期化
                swap_value = 0

                # open_date から current_date までの日付をループ
                current = open_date                
                while self.get_ny_business_date(current) < self.get_ny_business_date(current_date):
                    #date_str = current.astimezone(self.JP_TIMEZONE).strftime("%Y-%m-%d")  # 文字列に変換
                    date_str = self.get_tokyo_business_date(current).strftime("%Y-%m-%d")  # 文字列に変換、currentは日本時間とする
                    swap_value += data.get(date_str, {}).get('buy' if "Buy" in position else 'sell', 0)
                    print(f"date_str: {date_str}, swap_value:{swap_value}")
                    current += timedelta(days=1)  # 次の日に進める

                return swap_value * order_size / self.per_order_size    #2019年4月以降はoanda証券のサイトにあるデータはrollover込みの値なのでこれで良いが、それ以前はないので、スワップポイントを計算で求めないといけないので、rollover_daysを掛け合わせないといけない


    def get_ny_business_date(self,dt):
        
        #指定された日時に対して、7:00～翌日6:59の範囲で対応する基準日を返す。

        #Args:
        #    dt (datetime): 処理対象の日時（タイムゾーン付き）

        #Returns:
        #    datetime.date: 基準日の日付
        
        # 日本時間（JST）のタイムゾーンを設定
        #jst = pytz.timezone("Asia/Tokyo")
        try:
            dt = dt.astimezone(pytz.timezone('America/New_York'))
        except:
            pass

        # 入力日時を日本時間に変換（念のためタイムゾーンを揃える）
        if dt.tzinfo is None:
            raise ValueError("入力日時にはタイムゾーンが必要です。")
        #dt_jst = dt.astimezone(jst)

        # 時刻を判定して基準日を計算
        if dt.time() < self.NY_CLOSE_TIME:
            # 17:00未満の場合は前日が基準日
            reference_date = dt
        else:
            # 17:00以降の場合は当日が基準日
            reference_date = dt+timedelta(days=1)

        reference_date  =reference_date.date()

        return reference_date


    def get_tokyo_business_date(self,dt):
        
        #指定された日時に対して、7:00～翌日6:59の範囲で対応する基準日を返す。

        #Args:
        #    dt (datetime): 処理対象の日時（タイムゾーン付き）

        #Returns:
        #    datetime.date: 基準日の日付
        # 日本時間（JST）のタイムゾーンを設定
        #jst = pytz.timezone("Asia/Tokyo")


        dt_ny = dt.astimezone(pytz.timezone('America/New_York'))
        dt_jp = dt.astimezone(pytz.timezone('Asia/Tokyo'))

        diff_ny_jp = dt_ny.date() - dt_jp.date()
 
        # 時刻を判定して基準日を計算
        if dt_ny.time() < self.NY_CLOSE_TIME:
            # 17:00未満の場合は前日が基準日
            reference_date = dt_ny
        else:
            # 17:00以降の場合は当日が基準日
            reference_date = dt_ny + timedelta(days=1)

        reference_date += diff_ny_jp
        reference_date = reference_date.astimezone(self.JP_TIMEZONE)
        reference_date  =reference_date.date()

        return reference_date

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

        try:
            target_start = pytz.utc.localize(target_start)
            target_end = pytz.utc.localize(target_end)
        except:
            pass
        
        
        # ファイル検索と条件に合致するファイルの選択
        found_file = None
        for filename in os.listdir(directory):
            if filename.startswith(f'kabu_oanda_swapscraping_{rename_pair}_from'):
                # ファイルの start と end 日付を抽出
                try:
                    file_start = datetime.strptime(filename.split('_from')[1].split('_to')[0], '%Y-%m-%d')
                    file_end = datetime.strptime(filename.split('_to')[1].split('.csv')[0], '%Y-%m-%d')
                    file_start = pytz.utc.localize(file_start)
                    file_end = pytz.utc.localize(file_end)
                    
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
            self.swap_points_dict = swap_data.set_index('date').to_dict('index')
        else:
            print(f"scrape_from_oanda({pair}, {start_date}, {end_date})")
            self.swap_points_dict = self.scrape_from_oanda(pair, start_date, end_date).set_index('date').to_dict('index')

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
        url = "https://www.oanda.jp/course/ty3/swap"
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

            # スクロール処理の追加
            driver.execute_script("arguments[0].scrollIntoView();", swap_table)

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
        #csv ファイルとして保存
        data = [value for value in all_data.values()]

        df = pd.DataFrame(data, index=dates)

        df.reset_index(inplace=True)
        df.rename(columns={'index': 'date'}, inplace=True)

        pair = pair.replace("/","")

        df.to_csv(f'./csv_dir/kabu_oanda_swapscraping_{pair}_from{actual_start_date}_to{actual_end_date}.csv', index=False, encoding='utf-8')
        # 取得したデータを返す
        return df

        
    



if __name__ == "__main__":
    order_size = 1000
    #"""
    pair = "USDJPY=X"

    #this time is utc
    # 日本時間のタイムゾーン設定
    jst = pytz.timezone("Asia/Tokyo")
    
    # 日本時間で日時を設定（タイムゾーンを正しく設定）
    NY_TIMEZONE = pytz.timezone("America/New_York") 
    JP_TIMEZONE = pytz.timezone("Asia/Tokyo")

    #print("ny_timezone and jp_timezone")
    #print(start_date.astimezone(NY_TIMEZONE))
    #print(start_date.astimezone(JP_TIMEZONE))
    #print(end_date.astimezone(NY_TIMEZONE))
    #print(end_date.astimezone(JP_TIMEZONE))
    #print("\n")

    start_date = jst.localize(datetime(2024, 9, 3, 0, 0))
    end_date = jst.localize(datetime(2024, 9, 26, 23, 59))
    calculator = SwapCalculator("oanda", pair, start_date, end_date,interval="M1")


    start_date = jst.localize(datetime(2024, 9, 18, 14, 2))
    end_date = jst.localize(datetime(2024, 9, 19, 14, 2))
    total_swap_points = calculator.get_total_swap_points(pair, "Buy", start_date, end_date, order_size, [])
    print(total_swap_points)
    #"""
    #import datetime as dt_library
    #start_date = datetime(2021,1,4,21,59,tzinfo=dt_library.timezone.utc)
    #end_date = datetime(2021,1,5,22,0,tzinfo=dt_library.timezone.utc)
    #pair = "EURGBP=X"
    #calculator = SwapCalculator("oanda", pair, start_date, end_date,interval="M1")
#
    #total_swap_points = calculator.get_total_swap_points(pair, "Buy-Forced-Closed", start_date, end_date, order_size, [])
    #print(total_swap_points)

    #nzd = pytz.timezone('Pacific/Auckland')
    #start_date = nzd.localize(datetime(2024,1,10,0,0))
    #end_date = nzd.localize(datetime(2024,1,11,0,0))
#
    #pair = "AUDNZD=X"
    #calculator = SwapCalculator("oanda", pair, start_date, end_date,interval="M1")
    #total_swap_points = calculator.get_total_swap_points(pair, "Buy-Forced-Closed", start_date, end_date, order_size, [])
    #print(total_swap_points)
#

    #a = calculator.get_total_swap_points('USDJPY=X', "Buy", datetime(2024, 6, 4), datetime(2024, 6, 10), order_size, [])
    #b = calculator.get_total_swap_points('USDJPY=X', "Buy", datetime(2024, 6, 11), datetime(2024, 6, 11), order_size, [])
    #c = calculator.get_total_swap_points('USDJPY=X', "Buy", datetime(2024, 6, 12), datetime(2024, 6, 12), order_size, [])
    #print(f"a, b, a+b+c: {a}, {b}, {c}, {a + b + c}")
