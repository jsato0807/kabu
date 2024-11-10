import pandas as pd
import yfinance as yf
from datetime import datetime
import os

def get_file_id(url):
    # 正規表現パターン
    pattern = r'/d/([a-zA-Z0-9_-]+)'
    
    # パターンにマッチする部分を抽出
    match = re.search(pattern, url)
    
    if match:
        return match.group(1)  # マッチした部分の最初のグループ（ファイルID）を返す
    else:
        return None

def fetch_currency_data(pair, start, end, interval,link=None):
    if link is None:
        # Yahoo Financeからデータを取得
        data = yf.download(pair, start=start, end=end, interval=interval)
        # DateにUTC 0時0分を追加 (既にdate部分は0時で保存されている)
        data.index = pd.to_datetime(data.index)
        data.index = data.index.tz_localize('UTC').tz_convert('UTC')  # UTCにローカライズ

        # UTCの0時0分を表示するためにインデックスを更新
        data.index = data.index + pd.Timedelta(hours=0)  # 明示的に0時0分を設定

        data = data['Close']
        print(f"Fetched data length: {len(data)}")
        print(data.head())

        return data
    
    else:
        directory = "./csv_dir"
        target_start = start
        target_end = end
        found_file = None
        rename_pair = pair.replace("=X","")
        rename_pair = rename_pair[:3] + "_" + rename_pair[3:]

        for filename in os.listdir(directory):
             if filename.startswith(f"{rename_pair}") and filename.endswith(f".csv"):
                try:
                    file_start = datetime.strptime(filename.split('_from')[1].split('_to')[0], '%Y-%m-%d')
                    file_end = datetime.strptime(filename.split('_to')[1].split(f'_{interval}')[0], '%Y-%m-%d')

                    # start_date と final_end がファイルの範囲内か確認
                    if file_start <= target_start and file_end >= target_end:
                        found_file = filename
                        break
                except ValueError:
                    continue  # 日付フォーマットが違うファイルは無視


        if found_file:
            print(f"Loading data from {found_file}")
            file_path = os.path.join(directory, found_file)
            df = pd.read_csv(file_path)

        else:
            file_id = get_file_id(link)
            url = f"https://drive.google.com/uc?id={file_id}"



            # Google Driveの共有リンクからファイルIDを取得
            # 共有リンクは通常この形式: https://drive.google.com/file/d/FILE_ID/view?usp=sharing

            pair = pair.replace('=X','')

            pair = pair[:3] + "_" + pair[3:]
            start = str(start).split(' ')[0]
            end = str(end).split(' ')[0]

            filename = '{}_from{}_to{}_{}.csv'.format(pair, start, end, interval)


            # CSVファイルをダウンロード
            gdown.download(url, filename, quiet=False)

            # pandasを使ってCSVファイルをDataFrameに読み込む
            df = pd.read_csv(filename)             

        # time列をdatetime型に変換
        df['time'] = pd.to_datetime(df['time'])

        # Date列をtime列の値に更新
        df['Date'] = df['time']

        # Dateをインデックスに設定し、time列を削除
        df.set_index('Date', inplace=True)
        df.drop(columns=['time'], inplace=True)
        df = df['close']
        df = get_data_range(df, start, end)
        print(f"Fetched data length: {len(df)}")

        #print(df.head())
        #exit()
        
        return df


def get_data_range(data, current_start, current_end):
    #pay attention to data type; we should change the data type of current_start and current_end to strftime.
    result = {}
    start_collecting = False
    current_start = pd.Timestamp(current_start).tz_localize('UTC')
    current_end = pd.Timestamp(current_end).tz_localize('UTC')
    for date, values in data.items():

        # 指定された開始日からデータの収集を開始
        if date == current_start:
            start_collecting = True
        if start_collecting:
            result[date] = values
        # 指定された終了日でループを終了
        if date == current_end:
            break

    # 辞書をSeriesに変換し、列名を'Close'に指定
    result = pd.Series(result, name='Close')
    result.index.name = 'Date'

    return result