from datetime import datetime, timedelta
import holidays

# 祝日を考慮する関数
def is_holiday(date):
    jp_holidays = holidays.Japan(years=date.year)
    return date in jp_holidays

# 2営業日後の日付を計算する関数
def add_business_days(start_date, num_days):
    current_date = start_date
    added_days = 0
    while added_days < num_days:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5 and not is_holiday(current_date):
            added_days += 1
    return current_date

# ロールオーバーの日数を計算する関数（修正版）
def calculate_rollover_days(open_date,current_date):

    if current_date + timedelta(days=2) == add_business_days(current_date,2) :
        rollover_days = current_date - open_date + timedelta(days=1)
    if current_date + timedelta(days=2) < add_business_days(current_date,2):
        rollover_days = add_business_days(current_date,2) - (current_date+timedelta(days=1))
    
    return rollover_days

# メイン処理の例
def main():
    # 例としての初期の日付と終了日付
    open_date = datetime(2024, 4, 30)  # 水曜日
    current_date = datetime(2024, 5, 1)  # 木曜日

    rollover_days = calculate_rollover_days(open_date,current_date)
    print(f"Roll over days: {rollover_days}")

if __name__ == '__main__':
    main()

