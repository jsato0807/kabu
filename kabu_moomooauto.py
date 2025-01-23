from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time

# ChromeDriverのパスを設定
chrome_service = Service('path_to_chromedriver')

# Chromeオプションの設定
chrome_options = Options()
chrome_options.add_argument("--headless")  # ヘッドレスモードで実行する場合

# ブラウザの起動
driver = webdriver.Chrome(service=chrome_service, options=chrome_options)

try:
    # moomoo証券のデモトレードページにアクセス
    driver.get('https://www.moomoo.com/jp/paper-trade/assets?_ga=2.183782886.1766729397.1717719675-GA1.1.GA1.2.762912400.1717719672&_gac=1.254476922.1717719840.CjwKCAjwvIWzBhAlEiwAHHWgvUCFRzsp-3A7QEl21TILYY812ePgSCx4c0zT5_ouE5q_XqCvlP9NFRoCVU8QAvD_BwE&global_content=%7B%22promote_id%22%3A13509,%22sub_promote_id%22%3A40%7D')

    # ページが読み込まれるまで待機
    wait = WebDriverWait(driver, 10)

    # ログインが必要な場合、ログインフォームを操作
    username_input = wait.until(EC.presence_of_element_located((By.ID, 'username'))) # ここでIDを確認する
    password_input = driver.find_element(By.ID, 'password') # ここでIDを確認する
    login_button = driver.find_element(By.ID, 'login-button') # ここでIDを確認する

    username_input.send_keys('your_username')
    password_input.send_keys('your_password')
    login_button.click()

    # デモトレードページに移動するまで待機
    wait.until(EC.presence_of_element_located((By.ID, 'demo-trade-page-id'))) # ここでIDを確認する

    # デモトレードの操作を自動化する
    # 例: 特定の株を購入
    stock_search = driver.find_element(By.ID, 'stock-search-input') # ここでIDを確認する
    stock_search.send_keys('AAPL')
    stock_search.send_keys(Keys.RETURN)

    # 株が表示されるのを待機
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'stock-result-item'))) # ここでクラス名を確認する

    # 株を選択して購入
    stock_result = driver.find_element(By.CLASS_NAME, 'stock-result-item') # ここでクラス名を確認する
    stock_result.click()

    buy_button = driver.find_element(By.ID, 'buy-button') # ここでIDを確認する
    buy_button.click()

    # 必要に応じて購入量や価格を設定
    quantity_input = driver.find_element(By.ID, 'quantity-input') # ここでIDを確認する
    quantity_input.clear()
    quantity_input.send_keys('10')

    confirm_button = driver.find_element(By.ID, 'confirm-button') # ここでIDを確認する
    confirm_button.click()

    # 結果を確認するために少し待機
    time.sleep(5)

finally:
    # ブラウザを閉じる
    driver.quit()
