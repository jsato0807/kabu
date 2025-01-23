import oandapyV20
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.accounts as accounts

# OANDAのAPIキーとアカウントIDを設定
api_key = "c2fad4cffcc5baabf88caeaf45c82d45-fe82c00081ebe4f61910e3160cce1e65"  # あなたのAPIキーをここに設定


# APIクライアントの初期化
client = oandapyV20.API(access_token=api_key,environment="live")

# アカウントIDリクエスト
account_request = accounts.AccountList()

try:
    # アカウントリストを取得
    response = client.request(account_request)
    for account in response['accounts']:
        account_id = account['id']
        print(f"Account ID: {account_id}")

except oandapyV20.exceptions.V20Error as err:
    print(f"Error: {err}")



#account_id = "YOUR_ACCOUNT_ID"  # あなたのアカウントIDをここに設定

# ポジション情報の取得
position_request = positions.OpenPositions(accountID=account_id)

try:
    # ポジション情報をリクエスト
    response = client.request(position_request)
    
    # 取得したポジション情報を確認
    for position in response['positions']:
        instrument = position['instrument']  # 通貨ペア
        long_swap = position['long']['financing']  # ロングポジションのスワップポイント
        short_swap = position['short']['financing']  # ショートポジションのスワップポイント

        print(f"Instrument: {instrument}")
        print(f"Long Position Swap (Financing): {long_swap}")
        print(f"Short Position Swap (Financing): {short_swap}")

except oandapyV20.exceptions.V20Error as err:
    print(f"Error: {err}")
