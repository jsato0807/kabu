from alpha_vantage.timeseries import TimeSeries
import pandas as pd

# Alpha VantageのAPIキーをセットする
api_key = '4A8BSH45TRYYRF1X.'  # Alpha VantageからAPIキーを取得してください

# タイムシリーズオブジェクトを作成する
ts = TimeSeries(key=api_key)

# 銘柄コードと時間足（'1min'で1分足）を指定してデータを取得する
symbol = 'AAPL'  # Appleの例
data, meta_data = ts.get_intraday(symbol=symbol, interval='1min')

# データを表示して確認
print(data.head())
