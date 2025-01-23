import yfinance as yf
import pandas as pd

def check_price_movement(ticker, interval, lookback_period, threshold):
    # データを取得 (過去10年間のデータ)
    data = yf.download(ticker, period="10y", interval=interval)
    
    # 終値の変化を計算
    data['Price_Change'] = data['Close'].diff(periods=lookback_period)
    
    # 指定された閾値を超える値動きがあるかどうかをチェック
    data['Significant_Move'] = data['Price_Change'].abs() >= threshold
    
    # 結果を表示
    significant_moves = data[data['Significant_Move']]
    
    if not significant_moves.empty:
        print(f"The price of {ticker} moved by at least {threshold} within {lookback_period} {interval} intervals at the following times:")
        print(significant_moves[['Close', 'Price_Change']])
    else:
        print(f"The price of {ticker} did not move by at least {threshold} within {lookback_period} {interval} intervals in the last 10 years.")

# 使用例
check_price_movement('AUDNZD=X', '1h', 24, 0.016)  # 24時間以内に0.016以上の値動きがあったかをチェック

check_price_movement('EURGBP=X', '1h', 24, 0.007)

check_price_movement('USDCAD=X', '1h', 24, 0.0152)
