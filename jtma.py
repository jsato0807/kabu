import requests
from bs4 import BeautifulSoup

# ターゲットURL
url = "https://www.jtma.biz/job/"

# HTTPリクエストを送信
response = requests.get(url)

# ステータスコードを確認
if response.status_code == 200:
    # HTMLを解析
    soup = BeautifulSoup(response.content, 'html.parser')

    # 社名を取得
    company_names = []
    for td in soup.find_all('td', class_='column-1'):
        company_names.append(td.get_text(strip=True))

    # 結果を表示
    for name in company_names:
        print(name)
else:
    print(f"Failed to retrieve data: {response.status_code}")
