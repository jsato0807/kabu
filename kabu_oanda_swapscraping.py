from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
import time
from webdriver_manager.chrome import ChromeDriverManager  # 追加


def scrape_from_oanda():
    # Chromeのオプションを設定
    options = Options()
    options.add_argument('--headless')  # ヘッドレスモード（ブラウザを表示せずに実行）

    # WebDriverを初期化
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    # スワップポイントの情報が掲載されているURL
    url = "https://www.oanda.jp/course/ny4/swap"
    driver.get(url)

    # 通貨ペアを選択する（例: USD/JPY）
    currency_pair = Select(driver.find_element(By.CLASS_NAME, 'st-Select'))
    currency_pair.select_by_visible_text('USD/JPY')  # 必要な通貨ペアに変更

    # データ取得のための月を選択する（例: 2024年09月を選択）
    month_select = Select(driver.find_elements(By.CLASS_NAME, 'st-Select')[1])
    month_select.select_by_visible_text('2024年09月')  # 必要な月に変更

    # データを取得するために、少し待機
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
    scrape_from_oanda()
