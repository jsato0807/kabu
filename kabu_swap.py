from datetime import datetime, timedelta, time
import holidays
import requests
from bs4 import BeautifulSoup
import pytz

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
        'NZD': 'Pacific/Auckland',
        'AUD': 'Australia/Sydney',
        'EUR': 'Europe/Berlin',
        'GBP': 'Europe/London',
        'USD': 'America/New_York',
        'CAD': 'America/Toronto',
        'NOK': 'Europe/Oslo',
        'SEK': 'Europe/Stockholm',
    }
    def __init__(self, swap_points, pair, interval="1d"):
        self.swap_points_dict = self._create_swap_dict(swap_points)
        self.per_order_size = 10000 if pair in ['ZARJPY=X', 'MXNJPY=X'] else 1000 #MINKABU
        self.timezones = self.get_timezones_from_pair(pair)
        self.each_holidays = self.get_holidays_from_pair(pair)  # 祝日データをインスタンスに保持
        self.holiday_cache = {}  # 祝日判定結果のキャッシュ

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


    def get_timezones_from_pair(self, pair):
        currencies = pair.split('/')
        return [self.CURRENCY_TIMEZONES.get(currency) for currency in currencies]


    def get_holidays_from_pair(self, pair):
        # 通貨ペアから"X"を削除し、通貨コードを抽出
        pair = pair.replace("=X", "")
        base_currency = pair[:2]  # 最初の通貨
        quote_currency = pair[3:5]  # 次の通貨
        
        # 例外処理（ユーロなど）
        exceptions = {
            "EUR": "DE",   # ユーロはドイツを代表例に設定
            "XAU": None,   # 金は特定の国の祝日は不要
            "XAG": None    # 銀も同様
        }

        # 祝日を保持するHolidayBaseオブジェクト
        holidays_combined = holidays.HolidayBase()

        # ベース通貨とクオート通貨それぞれの祝日を追加
        for currency in [base_currency, quote_currency]:
            # 例外処理をチェック
            country_code = exceptions.get(currency, currency[:2])
            if country_code:  # Noneの場合はスキップ
                holidays_combined += holidays.CountryHoliday(country_code)

        return holidays_combined

    # 祝日をチェックするメソッド
    def is_holiday(self, date):
        # 入力が時刻を含むかどうかを判断
        for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d'):
            try:
                # 試行して日付を解析
                date = datetime.strptime(str(date), fmt).date()
                break  # 成功したらループを抜ける
            except ValueError:
                continue  # 失敗したら次の形式を試す

        if date not in self.holiday_cache:  # キャッシュに結果がない場合のみ計算
            self.holiday_cache[date] = date in self.each_holidays
        return self.holiday_cache[date]

    # 2営業日後の日付を計算するメソッド
    def add_business_days(self, start_date, num_units, trading_days_set, interval):
        # 現在の日時をニューヨーク時間に変換
        #start_date = start_date.astimezone(self.NY_TIMEZONE)

        current_date = start_date
        added_units = 0

        while added_units < num_units:
            #print(f"added_units:{added_units}")
            #print(f"business_day:{business_day}, utc:{business_day.astimezone(self.original_timezone)}")
            # timeframe_minutesを日数に変換

            current_date_before = current_date

            if interval == "1d":  # 日足の場合
                current_date += timedelta(days=1)
            elif interval == "H1":
                current_date += timedelta(hours=1)
            elif interval == "M1":
                current_date += timedelta(minutes=1)
            

            # 土日でなく、かつ祝日でない、もしくはtrading_daysに含まれている場合
            if (current_date.weekday() < 5 and 
                (not self.is_holiday(current_date.astimezone(pytz.timezone(self.timezones[0]))) and 
                 not self.is_holiday(current_date.astimezone(pytz.timezone(self.timezones[1])))) 
                 or current_date in trading_days_set):
                # NYクローズを跨いでいるかを判定
                if not self.crossed_ny_close(current_date_before.astimezone(self.NY_TIMEZONE)) and self.crossed_ny_close(current_date.astimezone(self.NY_TIMEZONE)) or interval == "1d":
                    added_units += 1

        #print(f"return of business_day:{business_day}")
        return current_date.astimezone(self.original_timezone)

    # NYクローズを跨いだかを判定するメソッド
    def crossed_ny_close(self, dt):

        # 夏時間・冬時間を考慮しつつ午後5時を跨いでいればTrueを返す
        if dt.time() >= self.NY_CLOSE_TIME:
            return True
        return False


    # ロールオーバーの日数を計算するメソッド
    def calculate_rollover_days(self, open_date, current_date, trading_days_set):
        try:
            rollover_days = (self.add_business_days(current_date, 2, trading_days_set, "1d") - self.add_business_days(open_date, 2, trading_days_set, "1d")).days


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
        rollover_days = self.calculate_rollover_days(open_date, current_date, trading_days_set)
        if pair not in self.swap_points_dict:
            return 0.0

        swap_value = self.swap_points_dict[pair].get('buy' if "Buy" in position else 'sell', 0)
        return swap_value * rollover_days * order_size / self.per_order_size

# URLからHTMLデータを取得する関数
def get_html(url):
    response = requests.get(url)
    response.raise_for_status()  # エラーが発生した場合、例外を発生させる
    return response.text

# 不換空白文字を削除する関数
def clean_text(text):
    return text.replace('\xa0', '').replace('円', '').strip()

# HTMLを解析してスワップポイントを抽出する関数
def parse_swap_points(html):
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
        swap_point = {headers[i]: clean_text(cells[i].get_text(strip=True)) for i in range(len(headers))}
        swap_points.append(swap_point)
    
    return swap_points

def rename_swap_points(swap_points):
    for item in swap_points:
        item['通貨名'] = convert_currency_name(item['通貨名'])
    return swap_points

# 通貨名を変換する関数
def convert_currency_name(currency_name):
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

if __name__ == "__main__":
    order_size = 1000
    url = 'https://fx.minkabu.jp/hikaku/moneysquare/spreadswap.html'
    html = get_html(url)
    swap_points = parse_swap_points(html)
    swap_points = rename_swap_points(swap_points)
    calculator = SwapCalculator(swap_points, 'USDJPY=X',interval="M1")
    
    total_swap_points = calculator.get_total_swap_points('USDJPY=X', "Buy", datetime(2024, 5, 28, 6, 1), datetime(2024, 9, 17, 6, 1), order_size, [])
    print(total_swap_points)

    #a = calculator.get_total_swap_points('USDJPY=X', "Buy", datetime(2024, 6, 4), datetime(2024, 6, 10), order_size, [])
    #b = calculator.get_total_swap_points('USDJPY=X', "Buy", datetime(2024, 6, 11), datetime(2024, 6, 11), order_size, [])
    #c = calculator.get_total_swap_points('USDJPY=X', "Buy", datetime(2024, 6, 12), datetime(2024, 6, 12), order_size, [])
    #print(f"a, b, a+b+c: {a}, {b}, {c}, {a + b + c}")
