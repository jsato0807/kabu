import yfinance as yf
import pandas as pd

# 例としてAppleの1時間足のデータを取得する
ticker = 'AAPL'
data = yf.download(ticker, interval='1h', start='2023-01-01', end='2023-12-31')

# データを表示して確認
print(len(data))

