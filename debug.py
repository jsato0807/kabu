import investpy

# 銘柄コードと期間を指定してデータを取得
data = investpy.get_stock_historical_data(stock="8766", country="japan", from_date="01/01/2023", to_date="31/03/2023")

# データの先頭を表示して確認
print(data.head())
