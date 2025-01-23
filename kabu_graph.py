import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

def plot_currency_data(ticker1, ticker2, start_date, end_date):
    # データを取得
    if ticker1:
        data1 = yf.download(ticker1, start=start_date, end=end_date)
    if ticker2:
        data2 = yf.download(ticker2, start=start_date, end=end_date)
    
    # 'Close'列をpipsに変換する関数
    def convert_to_pips(data, ticker):
        if 'JPY' in ticker:
            # JPYを含む通貨ペアは小数点2位
            return data * 100
        else:
            # JPYを含まない通貨ペアは小数点4位
            return data * 10000
    
    # pipsに変換
    data1_pips = convert_to_pips(data1['Close'], ticker1)
    data2_pips = convert_to_pips(data2['Close'], ticker2)
    print(f"ticker1:max{np.max(data1_pips)},min:{np.min(data1_pips)}")
    print(f"ticker2:max{np.max(data2_pips)},min:{np.min(data2_pips)}")
    
    # グラフの作成
    plt.figure(figsize=(12, 8))
    plt.scatter(data1.index, data1_pips, label=f'{ticker1} (pips)')
    plt.scatter(data2.index, data2_pips, label=f'{ticker2} (pips)')
    
    # グラフのタイトルとラベル
    plt.title(f'{ticker1} and {ticker2} Close Prices in Pips from {start_date} to {end_date}')
    plt.xlabel('Date')
    plt.ylabel('Change in Close Price (pips)')
    plt.legend()
    plt.grid(True)
    
    # グラフの表示
    plt.show()

# 使用例
ticker1 = 'USDJPY=X'  # 通貨ペア1
ticker2 = 'USDJPY=X'  # 通貨ペア2
start_date = '2019-04-04'
end_date = '2024-04-01'

plot_currency_data(ticker1, ticker2, start_date, end_date)
