""""
import backtrader as bt
import datetime

class GridTrading(bt.Strategy):
    params = (
        ('grid_size', 50),  # グリッドのサイズ（価格の間隔）
        ('grid_min', 1000), # グリッドの範囲（下限）
        ('grid_max', 2000), # グリッドの範囲（上限）
    )

    def __init__(self):
        self.buy_orders = []
        self.sell_orders = []
        self.grid_prices = []

        # グリッドの範囲内で注文価格を設定
        current_price = self.params.grid_min
        while current_price <= self.params.grid_max:
            self.grid_prices.append(current_price)
            current_price += self.params.grid_size

        self.order = None
        self.initial_cash = self.broker.get_cash()
        self.total_profit = 0

    def next(self):
        # 現在の価格を取得
        price = self.data.close[0]

        # グリッドの範囲内で買い注文と売り注文を発注
        for grid_price in self.grid_prices:
            if price <= grid_price and not self.position:  # 買い注文を発注
                self.order = self.buy(price=grid_price, exectype=bt.Order.Limit)
                print(f'BUY ORDER PLACED at {grid_price}')
            elif price >= grid_price + self.params.grid_size and self.position:  # 売り注文を発注
                self.order = self.sell(price=grid_price + self.params.grid_size, exectype=bt.Order.Limit)
                print(f'SELL ORDER PLACED at {grid_price + self.params.grid_size}')

        # 預託証拠金、有効証拠金、評価損益の表示
        cash = self.broker.get_cash()
        value = self.broker.get_value()
        position_value = self.position.size * self.data.close[0] if self.position else 0
        print(f'Cash: {cash:.2f}, Value: {value:.2f}, Position Value: {position_value:.2f}, Profit/Loss: {self.broker.get_value() - self.initial_cash:.2f}')

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f'BUY EXECUTED, Price: {order.executed.price}')
                # 決済用の売り注文を設定
                sell_price = order.executed.price + self.params.grid_size
                self.sell(price=sell_price, exectype=bt.Order.Limit)
                print(f'SELL ORDER PLACED at {sell_price}')
            elif order.issell():
                print(f'SELL EXECUTED, Price: {order.executed.price}')
                # 再度買い注文を設定
                buy_price = order.executed.price - self.params.grid_size
                if buy_price >= self.params.grid_min:
                    self.buy(price=buy_price, exectype=bt.Order.Limit)
                    print(f'BUY ORDER PLACED at {buy_price}')

    def notify_trade(self, trade):
        if trade.isclosed:
            print(f'TRADE PROFIT, GROSS {trade.pnl}, NET {trade.pnlcomm}')
            self.total_profit += trade.pnl

    def stop(self):
        # 確定利益の表示
        print(f'Total Profit: {self.total_profit:.2f}')
        print(f'Final Portfolio Value: {self.broker.getvalue():.2f}')

# データの読み込み（ここではYahoo Financeから取得したAAPLのデータを使用）
data = bt.feeds.YahooFinanceData(
    dataname='AAPL',
    fromdate=datetime.datetime(2020, 1, 1),
    todate=datetime.datetime(2023, 1, 1)
)

# Cerebroエンジンのセットアップ
cerebro = bt.Cerebro()
cerebro.addstrategy(GridTrading)
cerebro.adddata(data)
cerebro.broker.setcash(1000000)  # 初期資金1,000,000円
cerebro.addsizer(bt.sizers.FixedSize, stake=10)  # 取引サイズ

# バックテストの実行
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
cerebro.run()
print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())

# グラフの表示
cerebro.plot()
"""




import backtrader as bt
import datetime
import yfinance as yf

class GridTrading(bt.Strategy):
    params = (
        ('grid_size', 50),  # グリッドのサイズ（価格の間隔）
        ('grid_min', 100),  # グリッドの範囲（下限）
        ('grid_max', 200),  # グリッドの範囲（上限）
    )

    def __init__(self):
        self.buy_orders = []
        self.sell_orders = []
        self.grid_prices = []

        # グリッドの範囲内で注文価格を設定
        current_price = self.params.grid_min
        while current_price <= self.params.grid_max:
            self.grid_prices.append(current_price)
            current_price += self.params.grid_size

    def next(self):
        # 現在の価格を取得
        price = self.data.close[0]

        # グリッドの範囲内で買い注文と売り注文を発注
        for grid_price in self.grid_prices:
            if price <= grid_price and not self.position:  # 買い注文を発注
                self.buy(price=grid_price, exectype=bt.Order.Limit)
                print(f'BUY ORDER PLACED at {grid_price}')
            elif price >= grid_price + self.params.grid_size and self.position:  # 売り注文を発注
                self.sell(price=grid_price + self.params.grid_size, exectype=bt.Order.Limit)
                print(f'SELL ORDER PLACED at {grid_price + self.params.grid_size}')

""""
# データフレームからPandasDataに変換
class PandasData(bt.feeds.PandasData):
    params = (
        ('datetime', 'Date'),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume')
    )


# データの読み込み（ここではYahoo Financeから取得したAAPLのデータを使用）
data = yf.download("AMZN", start='2020-01-01', end='2023-01-01')

# Cerebroエンジンのセットアップ
cerebro = bt.Cerebro()
cerebro.addstrategy(GridTrading)
cerebro.adddata(PandasData(dataname=data))
cerebro.broker.setcash(1000000)  # 初期資金1,000,000円

# バックテストの実行
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
cerebro.run()
print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())

# グラフの表示
cerebro.plot()
"""


import backtrader as bt
import yfinance as yf
import pandas as pd
import datetime

# yfinanceでデータを取得
data = yf.download("AAPL", start='2020-01-01', end='2023-01-01')

# データフレームからPandasDataに変換
class PandasData(bt.feeds.PandasData):
    params = (
        ('datetime', 'date'),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume')
    )

# Cerebroエンジンのセットアップ
cerebro = bt.Cerebro()
cerebro.addstrategy(GridTrading)

# PandasDataでデータを追加
cerebro.adddata(PandasData(dataname=data))

cerebro.broker.setcash(1000000)  # 初期資金1,000,000円

# バックテストの実行
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
cerebro.run()
print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())

# グラフの表示
cerebro.plot()
