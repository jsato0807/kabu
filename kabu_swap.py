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
from kabu_compare_bis_intrestrate_and_oandascraping import Compare_Swap
from kabu_library import get_swap_points_dict, get_business_date
from kabu_oanda_swapscraping import arrange_pair_format
from functools import lru_cache
#from kabu_library import timing_decorator


#def timing_decorator(func):
#    def wrapper(*args, **kwargs):
#        start_time = time_module.time()
#        result = func(*args, **kwargs)
#        elapsed_time = time_module.time() - start_time
#        print(f"Method {func.__name__}: {elapsed_time:.6f} seconds")
#        return result
#    return wrapper


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
    #@timing_decorator
    def __init__(self, website, pair, start_date, end_date, order_size, trading_days_set):
        self.timezones = self.get_timezones_from_pair(pair)
        self.website = website
        self.each_holidays = self.get_holidays_from_pair(pair,start_date,end_date)  # 祝日データをインスタンスに保持
        #self.business_days = self.generate_business_days(pair, start_date, end_date)
        self.pair = pair

        if website == "minkabu":
            self.per_order_size = 10000 if pair in ['ZARJPY=X', 'MXNJPY=X'] else 1000 #MINKABU
            self.fetcher = ScrapeFromMinkabu()
            currencies = arrange_pair_format(pair)
            if not currencies[1] == "JPY":
                self.unit_conversion_data = fetch_currency_data(currencies[1]+"JPY=X", start_date, end_date, "1d")
            else:
                self.unit_conversion_data = None
        elif website == "oanda":
            self.per_order_size = 100000 if pair in ['ZARJPY=X','HKDJPY=X'] else 10000
            self.fetcher = ScrapeFromOanda(pair,start_date,end_date)
        
        self.swap_points_dict = self.fetcher.swap_points_dict

        if website == "oanda":
            if start_date.astimezone(pytz.timezone("Asia/Tokyo")) < pytz.timezone("Asia/Tokyo").localize(datetime(2019,4,1)):
                compare_swap = Compare_Swap(pair,start_date,end_date,order_size,months_interval=1,window_size=30,cumulative_period=1,cumulative_unit="month",swap_points_dict=self.swap_points_dict)
                self.swap_points_dict_theory = compare_swap.calculate_theory_swap(start_date,end_date)
                # 辞書にnumber_of_daysを追加
                jst = pytz.timezone("Asia/Tokyo")
                for date_str, values in self.swap_points_dict_theory.items():
                    open_date = jst.localize(datetime.strptime(date_str, "%Y-%m-%d")).replace(hour=7)
                    current_date = open_date+timedelta(days=1)
                    number_of_days = self.calculate_rollover_days(open_date, current_date, trading_days_set)
                    values["number_of_days"] = number_of_days
                    values["sell"] *= number_of_days
                    values["buy"] *= number_of_days


    def get_timezones_from_pair(self, pair):
        currencies = arrange_pair_format(pair)
        return [self.CURRENCY_TIMEZONES.get(currency) for currency in currencies]

    def get_holidays_from_pair(self, pair,start_date,end_date):
        currencies = arrange_pair_format(pair)
        
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


    # 祝日をチェックするメソッド
    def is_holiday(self, date, currency):
        date = date.date()
        return date in self.each_holidays[currency]

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


    """
    def convert_ny_close_to_local(self, holiday_date, timezone_str):
        #ニューヨーク時間17:00を指定されたタイムゾーンに変換
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
        #祝日を特定し、各通貨の現地時間でholiday_time_rangesを生成
        holidays_dict = self.get_holidays_from_pair(pair, start_date, end_date)

        holiday_time_ranges = {}
        # holidays_dictのキー（通貨）があっても、空のリストがある場合には空のリストをセット
        currencies = arrange_pair_format(pair)
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
        currencies = arrange_pair_format(pair)
    
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
    """

    #@timing_decorator
    def add_business_days(self, start_date, num_units, trading_days_set, interval, swap_flag=True):
        pair = self.pair
        # 現在の日時をニューヨーク時間に変換
        #start_date = start_date.astimezone(self.NY_TIMEZONE)

        current_date = start_date

        # 開始日時が営業日でない場合、最も近い営業日後まで進める
        while not self.is_ny_business_day(current_date):
            current_date += pd.Timedelta("1d")

        added_units = 0


        while added_units < num_units:
            # timeframe_minutesを日数に変換

            current_date_before = current_date
            #print(current_date_before)

            if interval == "1d":  # 日足の場合
                current_date += timedelta(days=1)
            elif interval == "H1":
                current_date += timedelta(hours=1)
            elif interval == "M1":
                current_date += timedelta(minutes=1)

            
            # 土日でなく、かつ祝日でない、もしくはtrading_daysに含まれている場合
            if swap_flag:
                if (self.is_ny_business_day(current_date) and 
                    (not self.is_holiday(current_date.astimezone(pytz.timezone(self.timezones[0])),arrange_pair_format(pair)[0]) and 
                     not self.is_holiday(current_date.astimezone(pytz.timezone(self.timezones[1])),arrange_pair_format(pair)[1]))):
                    # NYクローズを跨いでいるかを判定
                    if not self.crossed_ny_close(current_date_before) and self.crossed_ny_close(current_date) or interval == "1d":
                        added_units += 1
            else:
                if (self.is_ny_business_day(current_date) and 
                    (not self.is_holiday(current_date.astimezone(pytz.timezone(self.timezones[0])),arrange_pair_format(pair)[0]) and 
                     not self.is_holiday(current_date.astimezone(pytz.timezone(self.timezones[1])),arrange_pair_format(pair)[1])) 
                     or current_date in trading_days_set):
                    # NYクローズを跨いでいるかを判定
                    if not self.crossed_ny_close(current_date_before) and self.crossed_ny_close(current_date) or interval == "1d":
                        added_units += 1

        return current_date

    # NYクローズを跨いだかを判定するメソッド
    def crossed_ny_close(self, dt):

        # 夏時間・冬時間を考慮しつつ午後5時を跨いでいればTrueを返す
        if dt.time() >= self.NY_CLOSE_TIME:
            return True
        return False



    # ロールオーバーの日数を計算するメソッド
    #@timing_decorator
    def calculate_rollover_days(self, open_date, current_date, trading_days_set):
        try:
            rollover_days = (self.add_business_days(current_date, 2, trading_days_set, "1d") - self.add_business_days(open_date, 2, trading_days_set, "1d")).days
                    
            open_date = open_date.astimezone(pytz.timezone("America/New_York"))
            current_date = current_date.astimezone(pytz.timezone("America/New_York"))
            open_time = open_date.time()
            current_time = current_date.time()

            if open_time <= self.NY_CLOSE_TIME <= current_time:
                #if open_time== self.NY_CLOSE_TIME == current_time:
                #    pass
                #elif open_time == self.NY_CLOSE_TIME and self.NY_CLOSE_TIME != current_time:
                #    pass
                if open_time != self.NY_CLOSE_TIME and self.NY_CLOSE_TIME == current_time:
                    rollover_days += 1
                    print("added 1 to rollover_days in 'if open_time != self.NY_CLOSE_TIME and self.NY_CLOSE_TIME == current_time' in 'if open_time <= self.NY_CLOSE_TIME <= current_time' ")
                elif open_time != self.NY_CLOSE_TIME and self.NY_CLOSE_TIME != current_time:
                    rollover_days += 1
                    print("added 1 to rollover_days in 'if open_time != self.NY_CLOSE_TIME and self.NY_CLOSE_TIME != current_time' in 'if open_time <= self.NY_CLOSE_TIME <= current_time' ")

            elif current_time <= self.NY_CLOSE_TIME <= open_time:
                #if current_time == self.NY_CLOSE_TIME == open_time:
                #    pass
                #elif current_time != self.NY_CLOSE_TIME and self.NY_CLOSE_TIME == open_time:    #open_time=17:00 current_time=16:59
                #    pass
                if current_time == self.NY_CLOSE_TIME and self.NY_CLOSE_TIME != open_time:    #open_time=17:01 current_time=17:00
                    rollover_days += 1
                    print("added 1 to rollover_days in 'if current_time == self.NY_CLOSE_TIME and self.NY_CLOSE_TIME != open_time' in 'elif current_time <= self.NY_CLOSE_TIME <= open_time' ")
                #elif current_time != self.NY_CLOSE_TIME and self.NY_CLOSE_TIME != open_time:    #open_time=17:02 current_time=16:59
                #    pass

            elif  current_time <= open_time <= self.NY_CLOSE_TIME:
                #if current_time == open_time == self.NY_CLOSE_TIME:
                #    pass
                #elif current_time != open_time and open_time == self.NY_CLOSE_TIME: #open_time=17:00 current_time=16:59
                #    pass
                #elif current_time == open_time and open_time != self.NY_CLOSE_TIME: #open_time=16:59 current_time=16:59
                #    pass
                if current_time != open_time and open_time != self.NY_CLOSE_TIME: #open_time=16:59 current_time=16:58
                    rollover_days += 1
                    print("added 1 to rollover_days in 'elif current_time != open_time and open_time != self.NY_CLOSE_TIME' in 'elif current_time <= open_time <= self.NY_CLOSE_TIME' ")

            elif open_time <= current_time <= self.NY_CLOSE_TIME:
                #if open_time == current_time == self.NY_CLOSE_TIME:
                #    pass
                if open_time != current_time and current_time == self.NY_CLOSE_TIME:  #open_time=16:59 current_time=17:00
                    rollover_days += 1
                    print("added 1 to rollover_days in 'if open_time != current_time and current_time == self.NY_CLOSE_TIME' in 'elif open_time <= current_time <= self.NY_CLOSE_TIME' ")
                #elif open_time == current_time and current_time != self.NY_CLOSE_TIME:  #open_time=16:59 current_time=16:59
                #    pass
                #elif open_time != current_time and current_time != self.NY_CLOSE_TIME:  #open_time=16:58 current_time=16:59
                #    pass
                

            #elif self.NY_CLOSE_TIME <= open_time <= current_time:
            #    if self.NY_CLOSE_TIME == open_time == current_time:
            #        pass
            #    elif self.NY_CLOSE_TIME != open_time and open_time == current_time: #open_time=17:01 current_time=17:01
            #        pass
            #    elif self.NY_CLOSE_TIME == open_time and open_time != current_time: #open_time=17:00 current_time=17:01
            #        pass
            #    elif self.NY_CLOSE_TIME != open_time and open_time != current_time: #open_time=17:01 current_time=17:02
            #        pass


            elif self.NY_CLOSE_TIME <= current_time <= open_time:
                #if self.NY_CLOSE_TIME == current_time == open_time:
                #    pass
                #elif self.NY_CLOSE_TIME != current_time and current_time == open_time:  #open_time=17:01 current_time=17:01
                #    pass
                if self.NY_CLOSE_TIME == current_time and current_time != open_time:  #open_time=17:01 current_time=17:00
                    rollover_days += 1
                    print("added 1 to rollover_days in 'elif self.NY_CLOSE_TIME == current_time and current_time != open_time' in 'elif self.NY_CLOSE_TIME <= current_time <= open_time' ")
                elif self.NY_CLOSE_TIME != current_time and current_time != open_time:  #open_time=17:02 current_time=17:01
                    rollover_days += 1
                    print("added 1 to rollover_days in 'self.NY_CLOSE_TIME != current_time and current_time != open_time' in 'elif self.NY_CLOSE_TIME <= current_time <= open_time' ")

            
            
        except IndexError as e:
            print(f"Error in calculating rollover days: {e}")
            return 0
        return rollover_days

    #@timing_decorator
    def get_total_swap_points(self, pair, position, open_date, current_date, order_size, trading_days):
        if self.website == "minkabu":
            #trading_days_set = set(trading_days)
            rollover_days = self.calculate_rollover_days(open_date, current_date, trading_days)
            if rollover_days == 0:
                return 0
            elif rollover_days >= 1:
                if pair not in self.swap_points_dict:
                    return 0.0
                swap_value = self.fetcher.swap_points_dict[pair].get('buy' if "Buy" in position else 'sell', 0)
                if not self.unit_conversion_data is None:
                    current_date_str = current_date.strftime("%Y-%m-%d")
                    swap_value /= self.unit_conversion_data[current_date_str]
                return swap_value * rollover_days * order_size / self.per_order_size

        if self.website == "oanda":
                data = self.swap_points_dict
                # swap_value の初期化
                swap_value = 0

                # open_date から current_date までの日付をループ
                current = open_date
                # リスト内包表現を使用して、日付ごとの swap_value を合計
                #swap_value = sum(
                #    data.get(current_jst.strftime("%Y-%m-%d"), {}).get('buy' if "Buy" in position else 'sell', 0) 
                #    if current_jst >= pytz.timezone("Asia/Tokyo").localize(datetime(2019,4,1)).date() 
                #    else self.swap_points_dict_theory.get(current_jst.strftime("%Y-%m-%d"), {}).get('buy' if "Buy" in position else 'sell', 0)
                #    for current_jst in (get_tokyo_business_date(current) for current in pd.date_range(start=open_date, end=current_date))
                #)
                while self.get_ny_business_date(current) < self.get_ny_business_date(current_date):
                    date_str = current.astimezone(self.JP_TIMEZONE).strftime("%Y-%m-%d")  # 文字列に変換
                    current_jst = get_business_date(current,pytz.timezone("Asia/Tokyo"))
                    date_str = current_jst.strftime("%Y-%m-%d")  # 文字列に変換、currentは日本時間とする
                    swap_value += data.get(date_str, {}).get('buy' if "Buy" in position else 'sell', 0) if current_jst >= pytz.timezone("Asia/Tokyo").localize(datetime(2019,4,1)).date() else self.swap_points_dict_theory.get(date_str, {}).get('buy' if "Buy" in position else 'sell', 0)
                    print(f"date_str: {date_str}, swap_value:{swap_value}")

                    current += timedelta(days=1)  # 次の日に進める

                return swap_value * order_size / self.per_order_size    #2019年4月以降はoanda証券のサイトにあるデータはrollover込みの値なのでこれで良いが、それ以前はないので、スワップポイントを計算で求めないといけないので、rollover_daysを掛け合わせないといけない

    ##@timing_decorator
    @lru_cache(maxsize=None)
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


    def crossover_ny_close(self,start_date,end_date):
        if (end_date - start_date).days >= 1:
            return True
        else:
            start_date = start_date.astimezone(pytz.timezone('America/New_York'))
            end_date  = end_date.astimezone(pytz.timezone('America/New_York'))

            if start_date.time() < time(17,0) and end_date.time() >= time(17,0):
                 return True
            else:
                 return False

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
        
        rename_pair = pair.replace("/", "")
        rename_pair = rename_pair.replace("=X","")
        self.rename_pair = rename_pair

        self.swap_points_dict = get_swap_points_dict(start_date,end_date,self.rename_pair)

        
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
        
    



if __name__ == "__main__":
    order_size = 1000
    """
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
    calculator = SwapCalculator("minkabu", pair, start_date, end_date)


    start_date = jst.localize(datetime(2024, 9, 18, 14, 2))
    end_date = jst.localize(datetime(2024, 9, 19, 14, 2))
    total_swap_points = calculator.get_total_swap_points(pair, "Buy", start_date, end_date, order_size, [])
    print(total_swap_points)
    """
    import datetime as dt_library
    start_date = datetime(2019,3,25,21,59,tzinfo=dt_library.timezone.utc)
    end_date = datetime(2019,4,5,22,0,tzinfo=dt_library.timezone.utc)
    pair = "EURGBP=X"
    calculator = SwapCalculator("oanda", pair, start_date, end_date, order_size,[])

    total_swap_points = calculator.get_total_swap_points(pair, "Buy-Forced-Closed", start_date, end_date, order_size, [])
    print(total_swap_points)

    #nzd = pytz.timezone('Pacific/Auckland')
    #start_date = nzd.localize(datetime(2024,1,10,0,0))
    #end_date = nzd.localize(datetime(2024,1,11,0,0))
#
    #pair = "AUDNZD=X"
    #calculator = SwapCalculator("oanda", pair, start_date, end_date)
    #total_swap_points = calculator.get_total_swap_points(pair, "Buy-Forced-Closed", start_date, end_date, order_size, [])
    #print(total_swap_points)
#

    #a = calculator.get_total_swap_points('USDJPY=X', "Buy", datetime(2024, 6, 4), datetime(2024, 6, 10), order_size, [])
    #b = calculator.get_total_swap_points('USDJPY=X', "Buy", datetime(2024, 6, 11), datetime(2024, 6, 11), order_size, [])
    #c = calculator.get_total_swap_points('USDJPY=X', "Buy", datetime(2024, 6, 12), datetime(2024, 6, 12), order_size, [])
    #print(f"a, b, a+b+c: {a}, {b}, {c}, {a + b + c}")
