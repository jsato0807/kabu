import re
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

    # データを保存するための辞書
    all_data = {}
    
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
    
    # 指定された日付範囲で日毎のデータを取得
    current_date = start
    while current_date <= end:
        # 年月のテキストを作成
        year_month = current_date.strftime("%Y年%m月")
        
        # 月の選択
        month_select = Select(driver.find_elements(By.CLASS_NAME, 'st-Select')[1])
        try:
            month_select.select_by_visible_text(year_month)  # 指定された年月を選択
        except:
            print(f"{year_month} のデータは存在しません。")
            current_date += timedelta(days=1)
            continue
        
        # データ取得のために少し待機
        time.sleep(2)
        
        # テーブルを取得
        try:
            swap_table = driver.find_element(By.CLASS_NAME, 'tr-SwapHistory_Table')
        except:
            print(f"{year_month} のテーブルが見つかりません。")
            current_date += timedelta(days=1)
            continue

        # テーブルの行を取得し、該当する日付の行からデータを取得
        rows = swap_table.find_elements(By.TAG_NAME, 'tr')
        for row in rows[1:]:  # 最初の行はヘッダーなのでスキップ
            # 日付の列とデータの列を取得
            date_col = row.find_element(By.TAG_NAME, 'th')
            data_cols = row.find_elements(By.TAG_NAME, 'td')

            print(date_col.text)
            
            
            # 日付がcurrent_dateと一致するか確認
            date_text = re.sub(r'（.*?）', '', date_col.text)  # 曜日を除去

            # 各項目を辞書に保存
            if len(data_cols) == 3:  # 期待するデータ数を確認
                sell_text = data_cols[0].text.strip()
                buy_text = data_cols[1].text.strip()
                days_text = data_cols[2].text.strip()

                # 日付から「月」と「日」を除外し、日付部分を整数として取得
                match = re.match(r'(\d{2})月(\d{2})日', date_text)
                if match:
                    day = int(match.group(2))  # 「日」の部分を整数に変換
                    date_str = f"{current_date.year}-{current_date.month:02}-{day:02}"  # 年月日形式に変換
                else:
                    print("日付形式が不正です:", date_str)
                    continue
                
                all_data[date_str] = {
                    'sell': float(sell_text) if sell_text else 0,
                    'buy': float(buy_text) if buy_text else 0,
                    'number_of_days': int(days_text) if days_text else 0
                }

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
    print(len(data))
    for date, info in data.items():
        print(f"{date}: {info}")
