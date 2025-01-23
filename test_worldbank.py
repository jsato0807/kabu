import requests

def get_interest_rate(country_code, indicator):
    # World Bank APIのエンドポイント
    url = f"http://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}?format=json"
    
    # データの取得
    response = requests.get(url)
    
    # JSON形式でデータを解析
    data = response.json()
    
    # データの表示
    if len(data) > 1:
        for item in data[1]:  # data[0]はメタデータ
            print(f"Year: {item['date']}, Interest Rate: {item['value']}")
    else:
        print("データが見つかりませんでした。")

# 使用例
get_interest_rate("JP", "FR.INR.LEND")  # 日本の貸出金利データ
