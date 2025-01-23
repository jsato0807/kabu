import requests
import json

# OANDA APIエンドポイントとトークン
OANDA_API_URL = "https://api-fxtrade.oanda.com/v3/accounts/001-009-12527404-001/instruments"
OANDA_API_TOKEN = "c2fad4cffcc5baabf88caeaf45c82d45-fe82c00081ebe4f61910e3160cce1e65"

# リクエストヘッダー
headers = {
    "Authorization": f"Bearer {OANDA_API_TOKEN}"
}

# 利用可能な銘柄を取得
def get_instruments():
    response = requests.get(OANDA_API_URL, headers=headers)
    if response.status_code == 200:
        instruments = response.json().get("instruments", [])
        for instrument in instruments:
            print(f"{instrument['name']}: {instrument['displayName']}")
    else:
        print("エラーが発生しました:", response.status_code)

#get_instruments()


# 全ての銘柄を取得し、全資産クラスを表示
def get_all_instruments():
    response = requests.get(OANDA_API_URL, headers=headers)
    if response.status_code == 200:
        instruments = response.json().get("instruments", [])
        for instrument in instruments:
            print(f"{instrument['name']}: {instrument['displayName']} ({instrument['type']})")
    else:
        print("エラーが発生しました:", response.status_code, response.text)

#get_all_instruments()

900023896
OANDA_COMMODITIES_URL = "https://api-fxtrade.oanda.com/v3/accounts/900023896/instruments"
def get_commodities():
    response = requests.get(OANDA_COMMODITIES_URL, headers=headers)
    if response.status_code == 200:
        commodities = response.json().get("instruments", [])
        for commodity in commodities:
            print(f"{commodity['name']}: {commodity['displayName']} ({commodity['type']})")
    else:
        print("エラーが発生しました:", response.status_code, response.text)

get_commodities()
