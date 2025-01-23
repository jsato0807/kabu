from pymt5 import dde_client

# DDEクライアントのインスタンスを作成し、MetaTrader 5との接続を確立する
client = dde_client.DDEClient()
client.connect()

# シンボルの価格を取得する例
symbol = "EURUSD"
price = client.request(symbol, "Bid")

# 結果を表示する
print(f"The bid price for {symbol} is: {price}")

# 接続を閉じる
client.close()

