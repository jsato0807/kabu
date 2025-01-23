import requests
from bs4 import BeautifulSoup

# スクレイピングするURL
url = 'https://example.com/quotes'

# ページのHTMLを取得
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# BidとAskの要素を特定して取得
bid_element = soup.find(id='bid_price')
ask_element = soup.find(id='ask_price')

# テキストとして価格を取得（適切に解析する必要があります）
bid_price = bid_element.text.strip() if bid_element else 'N/A'
ask_price = ask_element.text.strip() if ask_element else 'N/A'

# 結果を表示
print(f'Bid Price: {bid_price}')
print(f'Ask Price: {ask_price}')
