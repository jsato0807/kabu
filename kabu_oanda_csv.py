import pandas as pd
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
from dateutil import parser
from datetime import timezone, timedelta
import time

# OANDA APIの設定
api_token = "c2fad4cffcc5baabf88caeaf45c82d45-fe82c00081ebe4f61910e3160cce1e65"  # 自分のOANDA APIトークンに置き換えてください
client = oandapyV20.API(access_token=api_token, environment="live")

def get_business_days(start_date, end_date):
    # 営業日を取得する（祝日は適宜設定）
    business_days = pd.date_range(start=start_date, end=end_date, freq='B')
    return business_days

def fetch_data_from_oanda(instrument, start_date, end_date):
    business_days = get_business_days(start_date, end_date)
    all_data = []  # 全てのデータを格納するリスト
    
    for day in business_days:
        total_points = 1440  # 1日あたりのポイント数
        
        for i in range(0, total_points, 500):
            # iの値に基づいてfromとtoを計算
            from_time = day + pd.Timedelta(minutes=i)
            to_time = day + pd.Timedelta(minutes=min(i + 500, total_points) - 1)

            # OANDA APIのパラメータ設定
            params = {
                'granularity': 'M1',
                'from': from_time.isoformat() + 'Z',  # UTC形式
                'to': to_time.isoformat() + 'Z'         # UTC形式
            }

            r = instruments.InstrumentsCandles(instrument=instrument, params=params)

            try:
                client.request(r)
                data = r.response['candles']
                
                # データをリストに追加
                for candle in data:
                    # 必要な情報を抽出して辞書に格納
                    candle_info = {
                        'time': candle['time'],
                        'open': candle['mid']['o'],
                        'high': candle['mid']['h'],
                        'low': candle['mid']['l'],
                        'close': candle['mid']['c'],
                        'volume': candle['volume']
                    }
                    all_data.append(candle_info)

                print(f"Fetched data for {instrument} from {from_time} to {to_time}.")
            except Exception as e:
                print(f"Error fetching data for {instrument}: {e}")
                time.sleep(1)  # 一時的なエラーの場合はスリープを入れる

    # データをDataFrameに変換
    df = pd.DataFrame(all_data)

    # CSVファイルに保存
    df.to_csv(f"{instrument}_from{start_date}_to{end_date}_data.csv", index=False)
    print(f"Data saved to {instrument}_from{start_date}_to{end_date}_data.csv.")

# 使用例
fetch_data_from_oanda("AUD_NZD", "2019-05-01", "2024-10-05")
