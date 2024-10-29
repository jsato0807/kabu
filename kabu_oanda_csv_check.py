import pandas as pd
import glob
import os
from kabu_calculate_range_sukumi import generate_currency_pairs

def check_missing_data_in_csv(directory, instrument, start_date, end_date, interval):
    # 特定のファイル名パターンにマッチするCSVファイルを取得
    directory = os.path.expanduser(directory)
    pattern = f"{directory}/{instrument}_from{start_date}_to{end_date}_{interval}.csv"
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        # 通貨ペアの順番が逆の場合を考慮
        reversed_instrument = instrument.split('_')[1] + '_' + instrument.split('_')[0]
        pattern = f"{directory}/{reversed_instrument}_from{start_date}_to{end_date}_{interval}.csv"
        csv_files = glob.glob(pattern)

    print(pattern)
    print(csv_files)

    # 欠損データを保存するためのリスト
    missing_data_summary = []

    for csv_file in csv_files:
        # CSVファイルを読み込み
        df = pd.read_csv(csv_file)
        
        # 欠損値があるかどうかを確認
        if df.isnull().values.any():
            print(f"Missing data found in {csv_file}:")
            # 欠損データをまとめる
            missing_data_info = {"file": csv_file, "missing": {}}
            
            # 各列ごとに欠損値の位置を表示
            missing_data = df.isnull()
            for column in df.columns:
                missing_rows = missing_data[column][missing_data[column] == True].index.tolist()
                
                if missing_rows:
                    print(f" - Column '{column}' has missing data at rows: {missing_rows}")
                    missing_data_info["missing"][column] = missing_rows

            missing_data_summary.append(missing_data_info)
        else:
            print(f"No missing data in {csv_file}")

    return missing_data_summary

# 使用例
directory = "~/github/kabu_dir"
# instrument = "USD_JPY"  # 通貨ペアを指定
start_date = "1994-01-01"
end_date = "2024-10-26"
interval = "M1"

currencies = ['JPY', 'ZAR', 'MXN', 'TRY', 'NZD', 'AUD', 'EUR', 'GBP', 'USD', 'CAD', 'NOK', 'SEK']
instruments = generate_currency_pairs(currencies)

# 全体の欠損データを保持するためのリスト
overall_missing_data = []

for instrument in instruments:
    instrument = instrument.replace('=X', '')
    instrument = instrument[:3] + "_" + instrument[3:]
    print(instrument)
    missing_data_summary = check_missing_data_in_csv(directory, instrument, start_date, end_date, interval)
    
    # 全体の欠損データをまとめる
    overall_missing_data.extend(missing_data_summary)

# 最後に欠損データの概要を表示
if overall_missing_data:
    print("\nSummary of missing data:")
    for entry in overall_missing_data:
        print(f"File: {entry['file']}")
        for column, rows in entry["missing"].items():
            print(f" - Column '{column}' has missing data at rows: {rows}")
else:
    print("No missing data found in any CSV files.")
