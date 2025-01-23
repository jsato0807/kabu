import pytz
from datetime import datetime, timedelta
from holidays import CountryHoliday

class HolidayProcessor:
    @staticmethod
    def convert_ny_close_to_local(holiday_date, timezone_str):
        """
        ニューヨーク時間の17:00を基準にして、各国のタイムゾーンに変換した祝日時間を取得
        """
        # ニューヨーク時間の17:00を基準に設定
        ny_timezone = pytz.timezone('America/New_York')
        ny_close_time = ny_timezone.localize(datetime(holiday_date.year, holiday_date.month, holiday_date.day, 17, 0, 0))

        # 指定されたタイムゾーンに変換
        local_timezone = pytz.timezone(timezone_str)
        local_time = ny_close_time.astimezone(local_timezone)

        return local_time

    def arrange_pair_format(self, pair):
        """
        通貨ペアを基に通貨コードを抽出して返す
        """
        if "=X" in pair:
            currencies = pair.replace("=X", "")
        elif "_" in pair:
            currencies = pair.split("_")
        elif "/" in pair:
            currencies = pair.split('/')
        else:
            currencies = [pair[:3], pair[3:]]
        return currencies

    @staticmethod
    def get_holidays_for_currencies(currencies, start_date, end_date):
        """
        各国の祝日を取得し、それを通貨ごとにdictに格納する
        """
        # 祝日を保持する辞書
        holidays_dict = {}

        # 祝日を取得する年の範囲
        years = list(range(start_date.year, end_date.year + 1))

        for currency in currencies:
            country_code = currency[:2]  # 通貨コードから国コードを取得

            # 例外処理
            if currency == "EUR":  # ユーロの場合はドイツを例に取る
                country_code = "DE"
            elif currency in ["XAU", "XAG"]:  # 金や銀の場合は祝日を無視
                holidays_dict[currency] = []
                continue

            # 国別の祝日を取得
            country_holidays = CountryHoliday(country_code, years=years)

            # 祝日リストをフィルタリングして取得
            filtered_holidays = [
                holiday for holiday in country_holidays.keys() 
                if start_date.date() <= holiday <= end_date.date()  # datetime.dateに変換して比較
            ]

            # 祝日を格納
            holidays_dict[currency] = filtered_holidays

        return holidays_dict

    def get_holiday_times(self, pair, start_date, end_date):
        """
        通貨ペアに基づいて祝日情報を取得し、開始・終了時刻を表示形式にまとめる
        """
        # 通貨ペアを基に通貨コードを取得
        currencies = self.arrange_pair_format(pair)
        
        # 各通貨のタイムゾーン辞書を設定
        timezone_dict = {
            "USD": "America/New_York",      # アメリカドル
            "EUR": "Europe/Berlin",         # ユーロ（ドイツを代表）
            "JPY": "Asia/Tokyo",            # 日本円
            "GBP": "Europe/London",         # イギリスポンド
            "AUD": "Australia/Sydney",      # オーストラリアドル
            "NZD": "Pacific/Auckland",      # ニュージーランドドル
            "CAD": "America/Toronto",       # カナダドル
            "CHF": "Europe/Zurich",         # スイスフラン
            "CNY": "Asia/Shanghai",         # 中国人民元
            "SEK": "Europe/Stockholm",      # スウェーデンクローナ
            "NOK": "Europe/Oslo",           # ノルウェークローネ
            "SGD": "Asia/Singapore",        # シンガポールドル
            "HKD": "Asia/Hong_Kong",        # 香港ドル
            "KRW": "Asia/Seoul",            # 韓国ウォン
            "INR": "Asia/Kolkata",          # インドルピー
            "ZAR": "Africa/Johannesburg",   # 南アフリカランド
            "MXN": "America/Mexico_City",   # メキシコペソ
            "BRL": "America/Sao_Paulo",     # ブラジルレアル
            "RUB": "Europe/Moscow",         # ロシアルーブル
            "TRY": "Europe/Istanbul",       # トルコリラ
            "PLN": "Europe/Warsaw",         # ポーランドズロチ
            "CZK": "Europe/Prague",         # チェココルナ
            "HUF": "Europe/Budapest",       # ハンガリーフォリント
            "THB": "Asia/Bangkok"           # タイバーツ
        }


        # 各通貨の祝日を取得
        holidays_dict = self.get_holidays_for_currencies(currencies, start_date, end_date)

        # 祝日情報を開始・終了時刻付きでリスト化
        holiday_times = {}

        for currency, holiday_dates in holidays_dict.items():
            for holiday_date in holiday_dates:
                # 祝日の開始時刻（ニューヨーク時間17:00を基準に変換）
                start_time = self.convert_ny_close_to_local(holiday_date, timezone_dict.get(currency))

                # 翌日のニューヨーク時間17:00を現地時間に変換し、その1分前を終了時刻に設定
                next_day_ny_close = self.convert_ny_close_to_local(holiday_date + timedelta(days=1), timezone_dict.get(currency))
                end_time = next_day_ny_close - timedelta(minutes=1)

                # 祝日の名前を作成
                holiday_name = f"{currency} Holiday"
                holiday_times[holiday_name] = {
                    "start": start_time,
                    "end": end_time
                }

        return holiday_times


holiday_processor = HolidayProcessor()
pair = "AUD/NZD"  # 通貨ペア
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)

holiday_times = holiday_processor.get_holiday_times(pair, start_date, end_date)

# 結果を表示
for holiday, times in holiday_times.items():
    print(f"Holiday: {holiday} | Start: {times['start']} | End: {times['end']}")
