from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
import time
from webdriver_manager.chrome import ChromeDriverManager

def scrape_from_oanda(pair, date):
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
    
    # 日付（年月）を選択
    month_select = Select(driver.find_elements(By.CLASS_NAME, 'st-Select')[1])
    month_select.select_by_visible_text(date)  # 引数で指定された年月を選択
    
    # データ取得のために、少し待機
    time.sleep(2)  # データがロードされるのを待つ
    
    # テーブルを取得
    swap_table = driver.find_element(By.CLASS_NAME, 'tr-SwapHistory_Table')
    
    # 各行からデータを抽出して表示
    rows = swap_table.find_elements(By.TAG_NAME, 'tr')
    for row in rows[1:]:  # 最初の行はヘッダーなのでスキップ
        cols = row.find_elements(By.TAG_NAME, 'td')  # td要素を取得
        data = [col.text for col in cols]  # テキストを抽出
        print(data)
    
    # WebDriverを終了
    driver.quit()

if __name__ == "__main__":
    # 使用例
    scrape_from_oanda("USD/JPY", "2024年09月")
