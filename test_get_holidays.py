import pytz
from datetime import datetime, timedelta
import holidays
import pandas as pd

class HolidayProcessor:
    timezone_dict = {
        "USD": "America/New_York",
        "JPY": "Asia/Tokyo",
        "AUD": "Australia/Sydney",
        "NZD": "Pacific/Auckland",
        # 必要な通貨のタイムゾーンを追加
    }

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

    def get_holidays_from_pair(self, pair, start_date, end_date):
        """通貨ペアから祝日を取得"""
        currencies = self.arrange_pair_format(pair)

        # 例外処理（ユーロなど）
        exceptions = {
            "EUR": "DE",   # ユーロはドイツを代表例に設定
            "XAU": None,   # 金は特定の国の祝日は不要
            "XAG": None    # 銀も同様
        }

        # 祝日を保持する辞書
        holidays_dict = {}
        years = list(range(start_date.year, end_date.year + 1))

        # ベース通貨とクオート通貨それぞれの祝日を追加
        for currency in currencies:
            # 例外処理をチェック
            country_code = exceptions.get(currency, currency[:2])
            if country_code:  # Noneの場合はスキップ
                holidays_dict[currency] = sorted(holidays.CountryHoliday(country_code, years=years).keys())

        return holidays_dict

    def convert_ny_close_to_local(self, holiday_date, timezone_str):
        """ニューヨーク時間17:00を指定されたタイムゾーンに変換"""
        ny_timezone = pytz.timezone("America/New_York")
        local_timezone = pytz.timezone(timezone_str)

        ny_start = ny_timezone.localize(datetime.combine(holiday_date, datetime.min.time()) + timedelta(hours=17)) - timedelta(days=1)

        local_start = ny_start.astimezone(local_timezone)

        local_end = local_start + timedelta(hours=23, minutes=59)

        return local_start, local_end

    def get_holiday_times(self, pair, start_date, end_date):
        """祝日を特定し、各通貨の現地時間で表示"""
        holidays_dict = self.get_holidays_from_pair(pair, start_date, end_date)

        holiday_times = {}
        for currency, holidays in holidays_dict.items():
            timezone = self.timezone_dict.get(currency)

            for holiday_date in holidays:
                start_time, end_time = self.convert_ny_close_to_local(holiday_date, timezone)
                holiday_times[f"{currency} Holiday ({holiday_date})"] = {
                    "start": start_time,
                    "end": end_time
                }

        return holiday_times


from datetime import datetime, timedelta

import pytz
from datetime import datetime, timedelta
import pandas as pd

class BusinessDayCalculatorWithHolidayProcessor:
    NY_CLOSE_TIME = datetime.strptime("07:00:00", "%H:%M:%S").time()  # 日本時間でNYクローズ
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

    def __init__(self, holiday_processor, pair, start_date, end_date):
        """
        営業日計算クラスの初期化
        :param holiday_processor: HolidayProcessor のインスタンス
        :param pair: 通貨ペア (例: "USD/JPY")
        :param start_date: 開始日 (datetime.date)
        :param end_date: 終了日 (datetime.date)
        """
        self.holiday_processor = holiday_processor
        self.timezones = self.get_timezones_from_pair(pair)
        self.business_days = self.generate_business_days(pair, start_date, end_date)

    def arrange_pair_format(self, pair):
        if "=X" in pair:
            currencies = pair.replace("=X", "")
        if "_" in pair:
            currencies = currencies.split("_")
        if "/" in pair:
            currencies = pair.split('/')
        if not "_" in pair and not "/" in pair:
            currencies = [pair[:3], pair[3:]]
        return currencies

    def get_timezones_from_pair(self, pair):
        currencies = self.arrange_pair_format(pair)
        return [self.CURRENCY_TIMEZONES.get(currency) for currency in currencies]

    def is_ny_business_day(self, date):
        """
        ニューヨーク時間で営業日を判定
        :param date: datetime オブジェクト（日本時間基準）
        :return: 営業日なら True、非営業日なら False
        """
        if date.weekday() == 0 and date.time() < self.NY_CLOSE_TIME:  # 月曜日の7:00前
            return False
        elif date.weekday() == 0 and date.time() >= self.NY_CLOSE_TIME:
            return True
        elif date.weekday() in [1, 2, 3, 4]:  # 火〜金曜日
            return True
        elif date.weekday() == 5 and date.time() < self.NY_CLOSE_TIME:  # 土曜日の7:00前
            return True
        return False

    def generate_business_days(self, pair, start_date, end_date):
        """
        営業日をリスト形式で生成
        :param pair: 通貨ペア (例: "USD/JPY")
        :param start_date: 開始日 (datetime.date)
        :param end_date: 終了日 (datetime.date)
        :return: 営業日のリスト
        """
        # 通貨ペアの祝日を取得
        holiday_times = self.holiday_processor.get_holiday_times(pair, start_date, end_date)

        # 通貨ペアの2国分の祝日時間を分ける
        currencies = self.arrange_pair_format(pair)
        holiday_time_ranges = {
            currencies[0]: [
                (info["start"], info["end"]) for info in holiday_times.values() if currencies[0] in holiday_times.keys()
            ],
            currencies[1]: [
                (info["start"], info["end"]) for info in holiday_times.values() if currencies[1] in holiday_times.keys()
            ]
        }

        # 全体の日時範囲を作成（start_date から end_date まで）
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())

        # 全体の日時リスト（1分単位で全ての日付）
        all_dates = pd.date_range(start=start_datetime, end=end_datetime, freq='min').to_pydatetime().tolist()

        # 祝日と休日の日時を除外
        business_days = []
        for date in all_dates:
            # 祝日や休日の時間範囲に含まれていない場合
            if not any(holiday_start <= date.astimezone(pytz.timezone(self.timezones[0])) <= holiday_end for holiday_start, holiday_end in holiday_time_ranges[currencies[0]]):
                if not any(holiday_start <= date.astimezone(pytz.timezone(self.timezones[1])) <= holiday_end for holiday_start, holiday_end in holiday_time_ranges[currencies[1]]):

                    # ニューヨーク時間基準で営業日かつ祝日でない場合
                    if self.is_ny_business_day(date.astimezone(pytz.timezone('Asia/Tokyo'))):
                        business_days.append(date)

        return business_days

    def add_business_days(self, start_datetime, num_units, interval="1d"):
        """
        任意の営業日単位で日付を進める
        :param start_datetime: 開始日時
        :param add_unit: 進めたい営業日数
        :param interval: 進める単位（例："1d"は1日, "1h"は1時間）
        :return: 進めた営業日
        """
        current_datetime = start_datetime
        added_units = 0

        while added_units < num_units:
            current_datetime += pd.Timedelta(interval)

            # 次の日付が営業日であればカウント
            if current_datetime in self.business_days:
                added_units += 1

        return current_datetime



# 実行例
holiday_processor = HolidayProcessor()
pair = "AUD/NZD"  # 通貨ペア
start_date = datetime(2024, 1, 1,tzinfo=pytz.utc)
end_date = datetime(2024, 12, 31,tzinfo=pytz.utc)

holiday_times = holiday_processor.get_holiday_times(pair, start_date, end_date)

# 結果を表示
for holiday, times in holiday_times.items():
    print(f"Holiday: {holiday} | Start: {times['start']} | End: {times['end']}")


# 使用例
calculator = BusinessDayCalculatorWithHolidayProcessor(holiday_processor, pair, start_date, end_date)

start_datetime = datetime(2024, 11, 5, 9, 0)  # 任意の開始日時
new_datetime = calculator.add_business_days(start_datetime, 1, "1m")
print(f"新しい営業日: {new_datetime}")