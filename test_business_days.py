import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
import holidays

# 日本の祝日を考慮したカスタム営業日
jp_holidays = holidays.Japan()
custom_bd = CustomBusinessDay(holidays=jp_holidays)

# NYクローズ時間（午後5時）
NY_CLOSE_TIME = pd.Timedelta(hours=17)

def get_next_business_time(start_datetime, minutes_later):
    """
    指定された分後の営業日時を取得
    NYクローズを跨いでいれば、次の営業日を返す
    """
    # 指定された時間（分）を加算
    future_datetime = start_datetime + pd.Timedelta(minutes=minutes_later)

    # NYクローズ時間を越えているか確認
    # NYクローズ（午後5時）の時刻を基準に
    ny_close_today = start_datetime.replace(hour=17, minute=0, second=0, microsecond=0)

    # もしfuture_datetimeがNYクローズを越えている場合、次の営業日に進める
    if future_datetime > ny_close_today:
        future_datetime = pd.date_range(start=future_datetime, periods=1, freq=custom_bd)[0]

    # future_datetimeが営業日かどうかを確認
    if future_datetime not in pd.date_range(start=future_datetime, periods=1, freq=custom_bd):
        # 営業日でない場合、次の営業日に進める
        future_datetime = pd.date_range(start=future_datetime, periods=1, freq=custom_bd)[0]
    
    return future_datetime

# 使用例
start_datetime = pd.Timestamp('2024-11-22 06:00:00')  # 開始日時
minutes_later = 1440  # 150分後（2時間30分後）

next_business_time = get_next_business_time(start_datetime, minutes_later)

print(f"開始日時: {start_datetime}")
print(f"{minutes_later}分後の営業日時: {next_business_time}")
