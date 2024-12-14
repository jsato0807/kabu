import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, time
import os
import re
import gdown
import statistics
import pytz
from kabu_oanda_swapscraping import scrape_from_oanda

def get_file_id(url):
    # 正規表現パターン
    pattern = r'/d/([a-zA-Z0-9_-]+)'
    
    # パターンにマッチする部分を抽出
    match = re.search(pattern, url)
    
    if match:
        return match.group(1)  # マッチした部分の最初のグループ（ファイルID）を返す
    else:
        return None

def fetch_currency_data(pair, start, end, interval):
    if interval=="1d":
        # Yahoo Financeからデータを取得
        data = yf.download(pair, start=start, end=end, interval=interval)
        # DateにUTC 0時0分を追加 (既にdate部分は0時で保存されている)
        data.index = pd.to_datetime(data.index)
        data.index = data.index.tz_localize('UTC').tz_convert('UTC')  # UTCにローカライズ

        # UTCの0時0分を表示するためにインデックスを更新
        data.index = data.index + pd.Timedelta(hours=0)  # 明示的に0時0分を設定

        data = data['Close']
        print(f"Fetched data length: {len(data)}")

        print(data.tail())

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
            link="https://drive.google.com/file/d/1XQhYNS5Q72nEqCz9McF5yizFxhadAaRT/view?usp=drive_link"    #the link of AUDNZD
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

def modified_to_japan_datetime(date):
    #modify str to jst datetime and all but jst datetime also modify to jst datetime
    try:
        date = datetime.strptime(date, '%Y-%m-%d')
    except:
        pass
        
    try:
        date = date.astimezone(pytz.timezone('Asia/Tokyo'))
    except:
        pass

    return date

def modified_to_utc_datetime(date):
    #modify str to jst datetime and all but jst datetime also modify to jst datetime
    try:
        date = datetime.strptime(date, '%Y-%m-%d')
        date = pytz.utc.localize(date)
    except:
        pass
        
    try:
        date = date.astimezone(pytz.utc)
    except:
        pass

    return date


def get_tokyo_business_date(dt):
        
    #指定された日時に対して、7:00～翌日6:59の範囲で対応する基準日を返す。

    #Args:
    #    dt (datetime): 処理対象の日時（タイムゾーン付き）

    #Returns:
    #    datetime.date: 基準日の日付
    # 日本時間（JST）のタイムゾーンを設定
    #jst = pytz.timezone("Asia/Tokyo")


    dt_ny = dt.astimezone(pytz.timezone('America/New_York'))
    dt_jp = dt.astimezone(pytz.timezone('Asia/Tokyo'))

    diff_ny_jp = dt_ny.date() - dt_jp.date()
 
    # 時刻を判定して基準日を計算
    if dt_ny.time() < time(17,0):
        # 17:00未満の場合は前日が基準日
        reference_date = dt_ny
    else:
        # 17:00以降の場合は当日が基準日
        reference_date = dt_ny + timedelta(days=1)

    reference_date += diff_ny_jp
    reference_date = reference_date.astimezone(pytz.timezone("Asia/Tokyo"))
    reference_date  =reference_date.date()

    return reference_date


def get_swap_points_dict(start_date,end_date,rename_pair):
    directory = './csv_dir'

    target_start = modified_to_japan_datetime(start_date)
    target_end = modified_to_japan_datetime(end_date)

    # データ取得の制限を確認
    jst = pytz.timezone('Asia/Tokyo')
    if target_start < jst.localize(datetime(2019,4,1)):
        print("2019年4月以前のデータはないので、理論値計算のためにstart_date=2019-4-1, end_date=datetime.now(jst)とします.")
        target_start = jst.localize(datetime(2019,4,1))
        target_end = datetime.now(jst)
        
    # ファイル検索と条件に合致するファイルの選択
    found_file = None
    partial_overlap_files = []

    for filename in os.listdir(directory):
        if filename.startswith(f'kabu_oanda_swapscraping_{rename_pair}_from'):
            try:
                file_start = datetime.strptime(filename.split('_from')[1].split('_to')[0], '%Y-%m-%d')
                file_end = datetime.strptime(filename.split('_to')[1].split('.csv')[0], '%Y-%m-%d')
                file_start = jst.localize(file_start)
                file_end = jst.localize(file_end)

                # 完全包含
                if file_start <= target_start and file_end >= target_end:
                    found_file = filename
                    break
                # 部分的に重なるファイルを収集
                elif (file_start <= target_end and file_end >= target_start):
                    partial_overlap_files.append((filename, file_start, file_end))
            except ValueError:
                continue

    # ファイルを読み込みまたはスクレイピング
    if found_file:
        print(f"Loading data from {found_file}")
        file_path = os.path.join(directory, found_file)
        swap_data = pd.read_csv(file_path)
        return swap_data.set_index('date').to_dict('index')

    elif partial_overlap_files:
        print("Loading data from partial overlap files")
        combined_data = pd.DataFrame()
    
        for filename, file_start, file_end in partial_overlap_files:
            file_path = os.path.join(directory, filename)
            swap_data = pd.read_csv(file_path)
            # 範囲を指定して抽出
            overlap_start = max(file_start, target_start)
            overlap_end = min(file_end, target_end)
            filtered_data = swap_data[(swap_data['date'] >= overlap_start.isoformat()) & 
                                      (swap_data['date'] <= overlap_end.isoformat())]
            # 重複を削除する
            filtered_data = filtered_data.drop_duplicates(subset='date')

            # combined_dataに追加（重複する日付は追加されない）
            combined_data = pd.concat([combined_data, filtered_data], ignore_index=True)

        # すべてのデータがcombined_dataに集められた後、重複を削除する
        combined_data = combined_data.drop_duplicates(subset='date', keep='first')
            
        # 日付順に並べ替え
        combined_data = combined_data.sort_values(by='date').reset_index(drop=True)
        print(combined_data)
        exit()

        # combined_data['date']をdatetime型に変換し、日本時間 (Asia/Tokyo) のタイムゾーンを設定
        combined_data['date'] = pd.to_datetime(combined_data['date'])
        if combined_data['date'].dt.tz is None:  # タイムゾーンが付与されていない場合
            combined_data['date'] = combined_data['date'].dt.tz_localize('Asia/Tokyo')
        else:  # 既にタイムゾーンが付与されている場合
            combined_data['date'] = combined_data['date'].dt.tz_convert('Asia/Tokyo')


        # 不足分を計算
        # タイムゾーンの有無を確認して適切に処理
        if combined_data['date'].dt.tz is None:  # タイムゾーンが付与されていない場合
            existing_dates = combined_data['date'].dt.tz_localize('Asia/Tokyo')
        else:  # すでにタイムゾーンが付与されている場合
            existing_dates = combined_data['date'].dt.tz_convert('Asia/Tokyo')

        missing_ranges = []
        if target_start < existing_dates.min():
            missing_ranges.append((target_start, existing_dates.min() - pd.Timedelta(days=1)))
        if target_end > existing_dates.max():
            missing_ranges.append((existing_dates.max() + pd.Timedelta(days=1), target_end))

        # 不足分をスクレイピング
        for scrape_start, scrape_end in missing_ranges:
            print(f"Scraping data from {scrape_start} to {scrape_end}")
            scraped_data = scrape_from_oanda(rename_pair, scrape_start, scrape_end)
            combined_data = pd.concat([combined_data, scraped_data])

        # 最終結果を返す
        return combined_data.set_index('date').to_dict('index')


    else:
        print(f"scrape_from_oanda({rename_pair}, {start_date}, {end_date})")
        return scrape_from_oanda(rename_pair, start_date, end_date).set_index('date').to_dict('index')