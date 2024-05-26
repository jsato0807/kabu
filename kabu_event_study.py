import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from kabu import get_month_end_business_day
import yfinance as yf
from datetime import datetime
from pandas.tseries.offsets import CustomBusinessDay
from datetime import datetime, timedelta
from pandas.tseries.offsets import CustomBusinessDay
from workalendar.asia import Japan

def get_business_days_before_and_after(date, num_days):
    # 日本のカレンダーを取得
    jp_cal = Japan()
    
    # 営業日カレンダーに基づいて営業日オフセットを作成
    jp_bd = CustomBusinessDay(calendar=jp_cal)
    
    # 指定した日から指定した日数前後の営業日を取得
    business_day_before = pd.date_range(end=date, periods=num_days, freq=jp_bd)[0]
    business_day_after = pd.date_range(start=date, periods=num_days, freq=jp_bd)[-1]
    
    return business_day_before, business_day_after



if __name__ == "__main__":


  print("write code, start_year,end_year")
  code, start_year,end_year = input().split()
      
  file = glob.glob("./txt_dir/{}*.txt".format(code)) 
  month = file[0].split("-")[2]
  period = file[0].split("-")[3]
  period = period.split(".")[0]

  
  # データを読み込む（適切なデータソースから読み込む必要があります）
  # ここでは例としてダミーデータを使用します
  # ダミーデータの形式: Date, Close
  # Dateは日付、Closeは株価を表します
  
  # ダミーデータの作成
  #dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='B')
  #closes = np.random.normal(loc=100, scale=10, size=len(dates))
  #df = pd.DataFrame({'Date': dates, 'Close': closes})
  
  
  for year in range(int(start_year),int(end_year)+1):
  
    #file_path = "./xlsx_dir/{}-{}-{}-{}-pre_grad_output.xlsx".format(code, year, month, period)
    # テスト用の日付
    day = get_month_end_business_day(year, month)
    date = pd.to_datetime('{}-{}-{}'.format(year,month,day))

    # 60日前と60日後の営業日を取得
    business_days_before, business_days_after = get_business_days_before_and_after(date, int(period))
    

    start_date = datetime(int(year), int(month), 1)
    df = yf.download("{}.T".format(code), start=business_days_before, end=business_days_after)
    
    # インデックスからDataFrameに変換し、Dateがdf内に存在するようにする
    df.reset_index(inplace=True)
    

    # 配当金配布日を指定
    event_date = date
    
    # 配当金配布日の前後の期間を指定
    pre_event_period = int(period)  # 配当金配布日の前30日
    post_event_period = int(period) # 配当金配布日の後30日
    
    # 配当金配布日の前後の期間を抽出
    pre_event_data = df[df['Date'] <= event_date].tail(pre_event_period)
    post_event_data = df[df['Date'] > event_date].head(post_event_period)

    
    # 配当金配布日の前後の期間の平均株価を計算
    pre_event_mean_price = pre_event_data['Close'].mean()
    post_event_mean_price = post_event_data['Close'].mean()
    print(pre_event_mean_price)
    print(post_event_mean_price)
    
       # プロット
    plt.figure(figsize=(10, 6))
    #plt.plot(df['Date'], df['Close'], label='Close Price')
    plt.axvline(pd.to_datetime(event_date), color='r', linestyle='--', label='Event Date')
    
    # 配当金前のデータをプロット
    plt.scatter(pre_event_data['Date'], pre_event_data['Close'], color='g', label='Pre Dividend')
    # 配当金後のデータをプロット
    plt.scatter(post_event_data['Date'], post_event_data['Close'], color='b', label='Post Dividend', alpha=0.5)
      
    plt.title('Event Study Analysis')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.show()

  