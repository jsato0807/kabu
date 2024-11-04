import pandas as pd
import gdown

#https://data.bis.org/topics/CBPOL/data
#のexportをクリックし、Bulk downloadsを選択したあと、Centrarl bank policy rates(CSV, flat)を選択するとデータを獲得可能


def filter_country_data(country_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    filename = "WS_CBPOL_csv_flat.csv"
    file_id = "1q74UoB3Oby_1Ormlg-wny9n8ohG7awpR"
    url = f"https://drive.google.com/uc?id={file_id}"

    # CSVファイルを読み込む
    # gdown.download(url, filename, quiet=False)

    data = pd.read_csv(f"./csv_dir/{filename}", sep=',', low_memory=False)
    """
    指定した国のデータをフィルタリングし、日付と観測値のペアを取得する関数。

    Args:
        data (pd.DataFrame): 元のデータフレーム
        country_name (str): フィルタリングする国名
        start_date (str): データ取得開始日 (YYYY-MM-DD形式)
        end_date (str): データ取得終了日 (YYYY-MM-DD形式)
        
    Returns:
        pd.DataFrame: 日付と観測値のペアを含むデータフレーム
    """
    # 'REF_AREA:Reference area'から国名を含む行をフィルタリング
    if "_" in country_name:
        country_name = country_name.replace("_", " ")
    country_data = data[data['REF_AREA:Reference area'].str.endswith(country_name)].copy()  # コピーを作成

    # 日付を変換し、無効な日付を除外
    country_data.loc[:, 'TIME_PERIOD:Time period or range'] = pd.to_datetime(country_data['TIME_PERIOD:Time period or range'], errors='coerce')
    country_data = country_data.dropna(subset=['TIME_PERIOD:Time period or range'])

    # 指定した日付の範囲でフィルタリング
    country_data = country_data[(country_data['TIME_PERIOD:Time period or range'] >= start_date) &
                                (country_data['TIME_PERIOD:Time period or range'] <= end_date)]

    # 必要な列のみを選択
    result_data = country_data[['TIME_PERIOD:Time period or range', 'OBS_VALUE:Observation Value']]

    start_date = start_date.date()
    end_date = end_date.date()
    country_name = country_name.replace(" ","_")
    result_data.to_csv(f"./csv_dir/kabu_bis_intrestrate_{country_name}_from{start_date}_to{end_date}.csv")
    
    return result_data

if __name__ == "__main__":
    # 任意の国名（例: 'Japan'）
    country_name = 'United States'  # ここを任意の国に変更
    # 任意の日付範囲
    start_date = pd.to_datetime('1996-01-01')  # ここを変更
    end_date = pd.to_datetime('2000-12-31')    # ここを変更
    japan_data = filter_country_data(country_name, start_date, end_date)

    print(japan_data)
