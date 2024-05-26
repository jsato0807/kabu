import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from kabu import get_month_end_business_day
import yfinance as yf
from datetime import datetime
from pandas.tseries.offsets import CustomBusinessDay
from workalendar.asia import Japan

def get_month_of_devidend(code):
    # エクセルファイルのパスを指定
    file_path = 'kabu.xlsx'

    # エクセルファイルを読み込む
    df = pd.read_excel(file_path, sheet_name='Sheet1')  # 'Sheet1'は読み込みたいシート名

    # コードを文字列に変換してから行を選択
    selected_row = df[df['code'].astype(str) == str(code)]

    # コードが見つからない場合
    if selected_row.empty:
        return []

    # 選択した行から月の文字列を取得
    month_str = selected_row['month'].iloc[0]  # シリーズから文字列を取得

    # `,` で分割して月のリストに変換
    months = month_str.split(',')

    # 空白を削除して返す
    months = [month.strip() for month in months]

    return months

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
    print("write code, start_year, end_year, period")
    code, start_year, end_year, period = input().split()

    months = get_month_of_devidend(code)

    # 全てのグラフを表示するためのプロットの数を計算
    total_plots = (int(end_year) - int(start_year) + 1) * len(months)
    num_cols = 4
    num_rows = (total_plots + num_cols - 1) // num_cols  # 切り上げ

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    plot_index = 0

    for year in range(int(start_year), int(end_year) + 1):
        for month in months:
            day = get_month_end_business_day(year, month)
            date = pd.to_datetime('{}-{}-{}'.format(year, month, day))

            business_days_before, business_days_after = get_business_days_before_and_after(date, int(period))
            start_date = datetime(int(year), int(month), 1)
            df = yf.download("{}.T".format(code), start=business_days_before, end=business_days_after)

            df.reset_index(inplace=True)
            event_date = date
            pre_event_period = int(period)
            post_event_period = int(period)
            pre_event_data = df[df['Date'] <= event_date].tail(pre_event_period)
            post_event_data = df[df['Date'] > event_date].head(post_event_period)

            pre_event_mean_price = pre_event_data['Close'].mean()
            post_event_mean_price = post_event_data['Close'].mean()
            print(pre_event_mean_price)
            print(post_event_mean_price)

            ax = axes[plot_index]
            plot_index += 1

            ax.plot(df['Date'], df['Close'], label='Close Price')
            ax.axvline(pd.to_datetime(event_date), color='r', linestyle='--', label='Event Date')
            ax.scatter(pre_event_data['Date'], pre_event_data['Close'], color='g', label='Pre Dividend')
            ax.scatter(post_event_data['Date'], post_event_data['Close'], color='b', label='Post Dividend', alpha=0.5)
            ax.set_title('Year: {} Month: {}'.format(year, month))
            ax.set_xlabel('Date')
            ax.set_ylabel('Close Price')
            ax.legend()
            ax.grid(True)

    # 余分なサブプロットを非表示にする
    for i in range(plot_index, len(axes)):
        axes[i].axis('off')

    # 全体のタイトルを追加
    fig.suptitle('Event Study Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('./png_dir/{}-{}_{}_{}_eventstudy_combined.png'.format(code, start_year, end_year, period))
    plt.show()
