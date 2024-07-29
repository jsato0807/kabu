from datetime import datetime, timedelta
import holidays
import requests
from bs4 import BeautifulSoup

# 祝日を考慮する関数
def is_holiday(date):
    jp_holidays = holidays.Japan(years=date.year)
    return date in jp_holidays

# 2営業日後の日付を計算する関数
def add_business_days(start_date, num_days):
    business_day = start_date
    added_days = 0
    while added_days < num_days:
        business_day += timedelta(days=1)
        if business_day.weekday() < 5 and not is_holiday(business_day):
            added_days += 1
    return business_day

# ロールオーバーの日数を計算する関数（修正版）
def calculate_rollover_days(open_date, current_date):
    rollover_days = (add_business_days(current_date, 2) - add_business_days(open_date, 2)).days
    return rollover_days




# URLからHTMLデータを取得する関数
def get_html(url):
    response = requests.get(url)
    response.raise_for_status()  # エラーが発生した場合、例外を発生させる
    return response.text

# 不換空白文字を削除する関数
def clean_text(text):
    return text.replace('\xa0', '').replace('円', '').strip()



# HTMLを解析してスワップポイントを抽出する関数
def parse_swap_points(html):
    soup = BeautifulSoup(html, 'html.parser')
    swap_table = soup.find('table')  # スワップポイントのテーブルを特定
    if not swap_table:
        raise ValueError("スワップポイントのテーブルが見つかりません")
    
    #print("Debug: Found table")  # デバッグ出力
    rows = swap_table.find_all('tr')
    
    # テーブルのヘッダーを取得
    headers = []
    for th in rows[0].find_all('th'):
        header_text = clean_text(th.get_text(strip=True))
        headers.append(header_text)

    #print(f"Debug: Headers found - {headers}")  # デバッグ出力

    # スワップポイントを格納するリスト
    swap_points = []

    # 各行のデータを抽出
    for row in rows[1:]:  # ヘッダー行を除外
        cells = row.find_all(['th', 'td'])
        if len(cells) != len(headers):
            print(f"Debug: Skipping incomplete row - {row}")  # デバッグ出力
            continue  # 不完全な行をスキップ
        swap_point = {headers[i]: clean_text(cells[i].get_text(strip=True)) for i in range(len(headers))}
        swap_points.append(swap_point)
    
    #print(f"Debug: Swap points found - {swap_points}")  # デバッグ出力
    return swap_points

def get_total_swap_points(swap_points,pair,position,open_date, current_date, order_size):
    
    rollover_days = calculate_rollover_days(open_date, current_date)
    #rollover_days = 0
    #print("rollover_days: {}".format(rollover_days))
  
    # 例として、全てのスワップポイントの合計を計算する（必要に応じて修正）
    total_swap_points = 0


    # 各要素の通貨名を変換する
    for item in swap_points:
        
        item['通貨名'] = convert_currency_name(item['通貨名'])

    for point in swap_points:
        if pair in point.values():

            try:
                if "Buy" in position:
                    buy_swap = float(point.get('買スワップ', 0))
                    if abs(buy_swap) > 0:  # 0以外の数値の場合にのみ加算する
                        total_swap_points += buy_swap
                if "Sell" in position:
                    sell_swap = float(point.get('売スワップ', 0))
                    if abs(sell_swap) > 0:  # 0以外の数値の場合にのみ加算する
                        total_swap_points += sell_swap
            except ValueError:
                print(f"Debug: Skipping invalid swap point value - {point.get('買スワップ')}")
                continue
            
    total_swap_points *= rollover_days * order_size/10000 # divided 1000 because this data is counted per 1000 Transaction currency amount
    
    return total_swap_points


# 通貨名を変換する関数
def convert_currency_name(currency_name):
    currency_mappings = {
    '米ドル/カナダドル': 'USDCAD=X',
    'ユーロ/米ドル': 'EURUSD=X',
    '英ポンド/米ドル': 'GBPUSD=X',
    '豪ドル/米ドル': 'AUDUSD=X',
    'NZドル/米ドル': 'NZDUSD=X',
    'ユーロ/英ポンド': 'EURGBP=X',
    '豪ドル/NZドル': 'AUDNZD=X',
    '米ドル/': 'USDJPY=X',
    'ユーロ/': 'EURJPY=X',
    '英ポンド/': 'GBPJPY=X',
    '豪ドル/': 'AUDJPY=X',
    'NZドル/': 'NZDJPY=X',
    'カナダドル/': 'CADJPY=X',
    '南アフリカランド/': 'ZARJPY=X',
    'トルコリラ/': 'TRYJPY=X',
    'メキシコペソ/': 'MXNJPY=X'
}
    for key, value in currency_mappings.items():
        if key in currency_name:
            return value
    return currency_name  # 変換できない場合はそのまま返す


if __name__ == "__main__":
    order_size = 1000
    url = 'https://fx.minkabu.jp/hikaku/moneysquare/spreadswap.html'
    html = get_html(url)
    #print(html[:1000])  # デバッグ出力：取得したHTMLの先頭部分を表示
    swap_points = parse_swap_points(html)
    total_swap_points =  get_total_swap_points(swap_points,'USDJPY=X',"Buy",datetime(2024,5,31),datetime(2024,6,12),order_size)
    print(total_swap_points)
    



    a = get_total_swap_points(swap_points,'USDJPY=X',"Buy",datetime(2024,6,4),datetime(2024,6,10),order_size)
    b = get_total_swap_points(swap_points,'USDJPY=X',"Buy",datetime(2024,6,11),datetime(2024,6,11),order_size)
    c = get_total_swap_points(swap_points,'USDJPY=X',"Buy",datetime(2024,6,12),datetime(2024,6,12),order_size)
    ###
    print(f"a,b,a+b+c:{a,b,c,a+b+c}")
