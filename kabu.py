import pandas as pd
import pandas_datareader.data as web
from datetime import datetime
import numpy as np
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import os

def get_month_end_business_day(year, month):
    # 指定した年月の初日を取得
    start_date = pd.to_datetime(f'{year}-{month}-01')
    
    # 営業月末を取得
    month_end_business_day = start_date + pd.offsets.BMonthEnd(1)
    return month_end_business_day.strftime('%d')

def get_gradient(df, year, month, day, period):
    # 特定の日付を指定
    target_date = '{}-{}-{}'.format(year, month, day)
    target_date = pd.to_datetime(target_date)

    # 日次リターンを計算
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
     
    # 前期間の株価データを抽出
    start_date_pre = target_date - pd.DateOffset(days=int(period))
    end_date_pre = target_date

    start_date_pre = start_date_pre.date()
    end_date_pre = end_date_pre.date()

    df_pre = df.loc[start_date_pre:end_date_pre]
    
    # 後期間の株価データを抽出
    start_date_post = target_date
    end_date_post = target_date + pd.DateOffset(days=int(period))

    start_date_post = start_date_post.date()
    end_date_post = end_date_post.date()

    df_post = df.loc[start_date_post:end_date_post]

    # 対数リターンの微分を計算
    df_pre['Log_Return_Diff'] = df_pre['Log_Return'].diff()
    pre_grad = df_pre['Log_Return_Diff']
    
    df_post['Log_Return_Diff'] = df_post['Log_Return'].diff()
    post_grad = df_post['Log_Return_Diff']    

    return pre_grad, post_grad

def write_directly(df, NAME, code, year, month, period):
    # Excelファイルを保存
    file_path = "./xlsx_dir/{}-{}-{}-{}-{}_output.xlsx".format(code, year, month, period, NAME)
    df.to_excel(file_path, index=True)  # インデックスを含めて書き込む

    print("{} を Excel ファイルに保存しました。".format(file_path))


if __name__ == '__main__':
    dt_now = datetime.now()

    # データ取得期間を設定
    start_date = datetime(1950, 1, 1)
    end_date = datetime(dt_now.year, dt_now.month, dt_now.day)

    # ユーザー入力を取得
    code, year, month, period = input().split("-")

    day = get_month_end_business_day(year, month)

    # 株価データを取得
    df = web.DataReader('{}.JP'.format(code), 'stooq', start_date, end_date).sort_index()
    #df = yf.download('{}.JP'.format(code), start_date, end_date).sort_index()

    # 特定の日付のデータを抽出
    specific_date = '{}-{}-{}'.format(year, month, day)
    if specific_date in df.index:
        specific_row = df.loc[specific_date]
        #print(specific_row)
    else:
        print(f"No data available for {specific_date}")
    
    pre_grad, post_grad = get_gradient(df, year, month, day, period)

    write_directly(pre_grad, "pre_grad", code, year, month, period)
    write_directly(post_grad, "post_grad", code, year, month, period)
