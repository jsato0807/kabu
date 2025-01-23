import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# 米国の金利データのティッカーシンボル
short_term_ticker_us = "^IRX"  # 13-week Treasury Bill
two_year_ticker_us = "^FVX"    # 2-Year Treasury Yield
ten_year_ticker_us = "^TNX"    # 10-Year Treasury Yield

# 日本の金利データのティッカーシンボル
two_year_ticker_jp = "^JP2YR"  # 2-Year Japan Government Bond Yield
ten_year_ticker_jp = "^JGB10"  # 10-Year Japan Government Bond Yield

# 開始日と終了日
start_date = "2000-01-01"
end_date = "2023-12-31"

# yfinanceを使ってデータをダウンロード
short_term_data_us = yf.download(short_term_ticker_us, start=start_date, end=end_date)
two_year_data_us = yf.download(two_year_ticker_us, start=start_date, end=end_date)
ten_year_data_us = yf.download(ten_year_ticker_us, start=start_date, end=end_date)
two_year_data_jp = yf.download(two_year_ticker_jp, start=start_date, end=end_date)
ten_year_data_jp = yf.download(ten_year_ticker_jp, start=start_date, end=end_date)

# プロットの準備
plt.figure(figsize=(14, 8))

# 米国の金利データのプロット
plt.plot(short_term_data_us.index, short_term_data_us['Close'], label='13-week Treasury Bill (US)')
plt.plot(two_year_data_us.index, two_year_data_us['Close'], label='2-Year Treasury Yield (US)')
plt.plot(ten_year_data_us.index, ten_year_data_us['Close'], label='10-Year Treasury Yield (US)')

# 日本の金利データのプロット
plt.plot(two_year_data_jp.index, two_year_data_jp['Close'], label='2-Year Japan Government Bond Yield')
plt.plot(ten_year_data_jp.index, ten_year_data_jp['Close'], label='10-Year Japan Government Bond Yield')

# グラフのタイトルとラベル
plt.title('US and Japan Interest Rates')
plt.xlabel('Date')
plt.ylabel('Yield (%)')
plt.legend()

# プロットを表示
plt.tight_layout()
plt.show()

