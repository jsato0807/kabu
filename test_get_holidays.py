import pytz
from datetime import datetime, timedelta
import holidays

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


# 実行例
holiday_processor = HolidayProcessor()
pair = "USD/JPY"  # 通貨ペア
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)

holiday_times = holiday_processor.get_holiday_times(pair, start_date, end_date)

# 結果を表示
for holiday, times in holiday_times.items():
    print(f"Holiday: {holiday} | Start: {times['start']} | End: {times['end']}")
