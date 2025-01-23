import yfinance as yf

# 例：Appleの株価データを取得
ticker = 'AAPL'

# Tickerオブジェクトを作成
stock = yf.Ticker(ticker)

# 1時間足のデータを取得（期間は過去7日間）
hourly_data = stock.history(period='5d', interval='1h')
print("Hourly Data:")
print(len(hourly_data))

# 1分足のデータを取得（期間は過去1日間）
minute_data = stock.history(period='5d', interval='1m')
print("Minute Data:")
print(len(minute_data))

