from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
from datetime import datetime, timedelta
import time
from webdriver_manager.chrome import ChromeDriverManager

def scrape_from_oanda(pair, start_date, end_date):
    # 日付をdatetime形式に変換
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    # データ取得の制限を確認
    if start < datetime(2019, 4, 1):
        print("2019年4月以前のデータはありません。")
        start = datetime(2019, 4, 1)

    # データを保存するためのリスト
    all_data = []
    
    # Chromeのオプションを設定
    options = Options()
    options.add_argument('--headless')  # ヘッドレスモード（ブラウザを表示せずに実行）
    
    # WebDriverを初期化
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    # スワップポイントの情報が掲載されているURL
    url = "https://www.oanda.jp/course/ny4/swap"
    driver.get(url)

    # 通貨ペアを選択
    currency_pair = Select(driver.find_element(By.CLASS_NAME, 'st-Select'))
    currency_pair.select_by_visible_text(pair)  # 引数で指定された通貨ペアを選択
    
    # 指定された日付範囲で月ごとのデータを取得
    current_date = start
    while current_date <= end:
        # 日付（年月）を文字列として取得
        year_month = current_date.strftime("%Y年%m月")
        
        # 日付を選択
        month_select = Select(driver.find_elements(By.CLASS_NAME, 'st-Select')[1])
        try:
            month_select.select_by_visible_text(year_month)  # 引数で指定された年月を選択
        except:
            print(f"{year_month} のデータは存在しません。")
            current_date += timedelta(days=31)
            current_date = current_date.replace(day=1)  # 次の月の1日に設定
            continue
        
        # データ取得のために少し待機
        time.sleep(2)  # データがロードされるのを待つ
        
        # テーブルを取得
        swap_table = driver.find_element(By.CLASS_NAME, 'tr-SwapHistory_Table')
        
        # 各行からデータを抽出して保存
        rows = swap_table.find_elements(By.TAG_NAME, 'tr')
        for row in rows[1:]:  # 最初の行はヘッダーなのでスキップ
            cols = row.find_elements(By.TAG_NAME, 'td')  # td要素を取得
            data = [col.text for col in cols]  # テキストを抽出
            all_data.append(data)  # データをリストに追加
        
        # 次の月へ進む
        current_date += timedelta(days=31)
        current_date = current_date.replace(day=1)  # 次の月の1日に設定

    # WebDriverを終了
    driver.quit()

    # 取得したデータを返す
    return all_data

# 使用例
if __name__ == "__main__":
    data = scrape_from_oanda("USD/JPY", "2021-01-01", "2021-03-31")
    for row in data:
        print(row)
