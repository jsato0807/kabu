import pandas as pd
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
from dateutil import parser
from datetime import timezone, timedelta, datetime
import time
from dateutil.relativedelta import relativedelta

# OANDA APIの設定
api_token = "c2fad4cffcc5baabf88caeaf45c82d45-fe82c00081ebe4f61910e3160cce1e65"  # 自分のOANDA APIトークンに置き換えてください
client = oandapyV20.API(access_token=api_token, environment="live")

def get_business_days(start_date, end_date):
    business_days = pd.date_range(start=start_date, end=end_date, freq='B')
    return business_days

def fetch_data_from_oanda(instrument, start_date, end_date, interval,instrument_swapped=False):
    business_days = get_business_days(start_date, end_date)
    all_data = []  # 全てのデータを格納するリスト
    
    for day in business_days:
        total_points = 1440  # 1日あたりのポイント数
        
        for i in range(0, total_points, 500):
            from_time = day + pd.Timedelta(minutes=i)
            to_time = day + pd.Timedelta(minutes=min(i + 500, total_points) - 1)

            params = {
                'granularity': f'{interval}',
                'from': from_time.isoformat() + 'Z',
                'to': to_time.isoformat() + 'Z'
            }

            r = instruments.InstrumentsCandles(instrument=instrument, params=params)

            attempt = 0
            success = False
            max_attempts = 1
            retry_delay = 5


            while not success and attempt < max_attempts and not instrument_swapped:
                try:
                    client.request(r)
                    data = r.response['candles']

                    for candle in data:
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
                    success = True
                    
                except Exception as e:
                    error_message = str(e)
                    print(f"Error fetching data for {instrument}: {error_message}")
                    
                    # 通貨ペアを逆順にする
                    if "Invalid value specified for 'instrument'" in error_message and not instrument_swapped:
                        instrument = instrument[4:] + "_" + instrument[:3]
                        instrument_swapped = True  # 通貨ペアを入れ替えたフラグを更新
                        print(f"Retrying with reversed instrument: {instrument}")
                    elif attempt >= max_attempts - 1:
                        # 通貨ペアを逆にしても失敗する場合、start_dateを1年後に進めて再試行
                        start_date = (parser.parse(start_date) + relativedelta(years=1)).strftime('%Y-%m-%d')
                        print(f"All retries failed. Advancing start_date to {start_date} and retrying.")
                        return fetch_data_from_oanda(instrument, start_date, end_date, interval)
                    else:
                        attempt += 1
                        print(f"Retrying in {retry_delay} seconds... (Attempt {attempt}/{max_attempts})")
                        time.sleep(retry_delay)

    df = pd.DataFrame(all_data)

    if not df.empty:
        df.to_csv(f"~/github/kabu_dir/csv_dir/{instrument}_from{start_date}_to{end_date}_{interval}.csv", index=False)
        print(f"Data saved to {instrument}_from{start_date}_to{end_date}_{interval}.csv.")
    else:
        print(f"No data available for {instrument} in the specified date range.")


def generate_currency_pairs(currencies, base_currency='USD'):
    """
    通貨リストから通貨ペアを生成し、USDを基軸として
    最小限の通貨ペアデータで他のペアも計算できるようにする関数。
    """
    pairs_to_download = []
    pairs_to_calculate = []
    
    # USD基軸の通貨ペアを生成してダウンロードリストに追加
    for currency in currencies:
        if currency != base_currency:
            pair = currency + base_currency + '=X'
            pairs_to_download.append(pair)
    
    # 基軸通貨を使って構成できるペアを計算リストに追加
    for i in range(len(currencies)):
        for j in range(len(currencies)):
            if i != j:
                pair = currencies[i] + currencies[j] + '=X'
                
                # 基軸通貨が含まれないペアを計算リストに追加
                if base_currency not in [currencies[i], currencies[j]]:
                    pairs_to_calculate.append(pair)
    
    return pairs_to_download, pairs_to_calculate


# 使用例
if __name__ == "__main__":
    #currencies = ['JPY','ZAR','MXN','TRY','CHF','NZD','AUD','EUR','GBP','USD','CAD','NOK','SEK']
    #currencies = ['JPY','XAU','BTC','WTI','XAG','ETH','LTI','XRP']
    #""""
    #currencies = ['USD','BTC','XAG','ETH','LTI','XRP']
    #currencies = ['USD','WTI']
    #currencies = ['USD','WTI']
    currencies = ['USD','XRP']

    pairs_to_download, pairs_to_calculate = generate_currency_pairs(currencies)

    # テスト例
    currency_pairs = pairs_to_download
    
    for currency_pair in currency_pairs:
        currency_pair = currency_pair.replace('=X','')
        instrument = currency_pair[:3] + "_" + currency_pair[3:]

        # データを取得
        print(instrument)
        fetch_data_from_oanda(instrument, "1994-01-01", "2024-10-26", "M1")
    #"""

    """
    Instruments = ['SPX500_USD','NAS100_USD','JP225_JPY']
    Instruments = ['NAS100_USD','JP225_JPY']
    for instrument in Instruments:
        fetch_data_from_oanda(instrument, "1994-01-01", "2024-10-26", "M1")
    """

