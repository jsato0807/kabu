import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from kabu import get_month_end_business_day
import yfinance as yf
from datetime import datetime
from pandas.tseries.offsets import CustomBusinessDay
from workalendar.asia import Japan
from kabu_event_study import extract_year_month

def get_month_of_devidend(code):
    file_path = 'kabu.xlsx'
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    selected_row = df[df['code'].astype(str) == str(code)]
    if selected_row.empty:
        return []
    month_str = selected_row['month'].iloc[0]
    months = month_str.split(',')
    months = [month.strip() for month in months]
    return months

def get_business_days_before_and_after(date, num_days):
    jp_cal = Japan()
    jp_bd = CustomBusinessDay(calendar=jp_cal)
    business_day_before = pd.date_range(end=date, periods=num_days, freq=jp_bd)[0]
    business_day_after = pd.date_range(start=date, periods=num_days, freq=jp_bd)[-1]
    return business_day_before, business_day_after

if __name__ == "__main__":
    print("write code,year,month,period")
    code, year, month, period = input().split("-")

    nikkei = yf.Ticker("^N225").history(period="max")
    
    # nikkei のインデックスを tz-naive にする
    nikkei.index = nikkei.index.tz_localize(None)

    filenames = glob('./txt_dir/{}*{}*.txt'.format(code, period))
    sorted_files = sorted(filenames, key=lambda x: extract_year_month(x))
    
    start_year = sorted_files[0].split("-")[1]
    end_year = sorted_files[-1].split("-")[1]
    
    months = get_month_of_devidend(code)

    total_plots = (int(end_year) - int(start_year) + 1) * len(months)
    num_cols = 4
    num_rows = (total_plots + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    plot_index = 0

    for year in range(int(start_year), int(end_year) + 1):
        for month in months:
            day = get_month_end_business_day(year, month)
            date = pd.to_datetime('{}-{}-{}'.format(year, month, day))

            business_days_before, business_days_after = get_business_days_before_and_after(date, int(period))

            # business_days_before および business_days_after を tz-naive にする
            business_days_before = business_days_before.tz_localize(None)
            business_days_after = business_days_after.tz_localize(None)

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

            # x軸の日付表示を斜めにする
            ax.tick_params(axis='x', rotation=45)

            # 第2軸を追加して日経225をプロット
            ax2 = ax.twinx()
            nikkei_filtered = nikkei.loc[business_days_before:business_days_after]
            ax2.plot(nikkei_filtered.index, nikkei_filtered['Close'], label="Nikkei 225", color='orange')

            ax.set_title('Year: {} Month: {}'.format(year, month))
            ax.set_xlabel('Date')
            ax.set_ylabel('Close Price')
            ax2.set_ylabel('Nikkei 225')

            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.grid(True)

    for i in range(plot_index, len(axes)):
        axes[i].axis('off')

    fig.suptitle('Event Study Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('./png_dir/{}-{}_{}_{}_eventstudy_combined.png'.format(code, start_year, end_year, period))
    print("saved ./png_dir/{}-{}_{}_{}_eventstudy_combined.png".format(code, start_year, end_year, period))
