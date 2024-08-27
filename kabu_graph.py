import yfinance as yf
import matplotlib.pyplot as plt

def plot_currency_data(ticker, start_date, end_date):
    # データを取得
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # グラフの作成
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'], label='Close Price')
    
    # グラフのタイトルとラベル
    plt.title(f'{ticker} Close Price from {start_date} to {end_date}')
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)')
    plt.legend()
    plt.grid(True)
    
    # グラフの表示
    plt.show()

# 使用例
ticker = 'AUDNZD=X'  # 為替レートのティッカーシンボル
start_date = '2019-05-01'
end_date = '2024-05-01'

plot_currency_data(ticker, start_date, end_date)
