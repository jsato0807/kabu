import requests
import pandas as pd

# Alpha Vantage APIのエンドポイントとAPIキー
api_key = 'IHCKN3RZH4A8IIU1'
url = 'https://www.alphavantage.co/query'

# APIリクエストパラメータ
params = {
    'function': 'FX_INTRADAY',
    'from_symbol': 'USD',
    'to_symbol': 'JPY',
    'interval': '60min',  # 1時間ごとのデータ
    'apikey': api_key
}

# APIリクエストの送信
response = requests.get(url, params=params)
data = response.json()

# レスポンスに基づくデータフレームの作成
if 'Time Series FX (60min)' in data:
    df = pd.DataFrame.from_dict(data['Time Series FX (60min)'], orient='index')
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    print(df.head())
else:
    print("Error:", data)
