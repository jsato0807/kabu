import oandapyV20
import oandapyV20.endpoints.instruments as instruments
from oandapyV20.exceptions import V20Error
import pandas as pd

# OANDA API接続情報
api_token = "c2fad4cffcc5baabf88caeaf45c82d45-fe82c00081ebe4f61910e3160cce1e65"  # OANDAのAPIトークンを設定
#account_id = "001-009-12527404-001"  # OANDAのアカウントIDを設定

client = oandapyV20.API(access_token=api_token, environment="live")

def get_oanda_data(instrument, granularity, count=100):
    """
    OANDAから指定の通貨ペアのヒストリカルデータを取得する関数。
    
    Parameters:
    instrument (str): 通貨ペア（例: "EUR_USD"）
    granularity (str): 時間足（例: "M1", "H1", "D"など）
    count (int): 取得するデータの数（デフォルトは100）
    
    Returns:
    pandas.Series: 取得したデータをSeries形式で返す
    """
    params = {
        "granularity": granularity,
        "count": count
    }
    
    # APIリクエスト
    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    
    try:
        client.request(r)
        data = r.response['candles']
        
        # データフレーム化
        df = pd.DataFrame([{
            'time': candle['time'],
            'close': float(candle['mid']['c'])
        } for candle in data])
        
        # timeをdatetime形式に変換し、timeをインデックスに設定
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        # closeカラムのみのSeriesに変換
        series = df['close']
        
        return series
    
    except V20Error as err:
        print(f"Error: {err}")
        return None

# 例: EUR/USDの1分足データを100件取得
instrument = "EUR_USD"
granularity = "M1"  # 1分足 (M1), 1時間足 (H1), 日足 (D)などを指定
data = get_oanda_data(instrument, granularity, count=100)

# データの表示
print(type(data))  # Series型か確認
print(data)
