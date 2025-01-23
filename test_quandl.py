import quandl
import requests

#quandl.ApiConfig.api_key =  'eymPWwa6RdMvpANN_bi1'
API_KEY = 'szHTEHZPWsAv8UM_HwZS'
quandl.ApiConfig.api_key =  API_KEY



def list_datasets(database_code):
    try:
        # APIリクエストを送信してデータセットを取得
        url = f"https://www.quandl.com/api/v3/databases/{database_code}?api_key={API_KEY}"
        response = requests.get(url)
        
        # レスポンスのステータスコードを確認
        if response.status_code != 200:
            print(f"Error: Unable to fetch data from Quandl. Status code: {response.status_code}")
            print(response.json())
            return
        
        data = response.json()

        # データベースの情報を表示
        if 'database' in data:
            print(f"Database: {data['database']['name']}\nAvailable Datasets:")
            
            # 各データセットの情報を表示
            for dataset in data['database']['datasets']:
                print(f" - {dataset['code']}: {dataset['name']}")
        else:
            print("Error: 'database' key not found in the response.")
            print(data)  # エラーメッセージを表示
    except Exception as e:
        print(f"An error occurred: {e}")

# 例として、ニュージーランドのデータベースを指定
list_datasets('AUD')  # RBNZ（ニュージーランド準備銀行）のデータセットを表示

