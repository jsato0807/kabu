from datetime import datetime, timedelta
import holidays
import requests
from bs4 import BeautifulSoup

# 祝日を考慮する関数
def is_holiday(date):
    jp_holidays = holidays.Japan(years=date.year)
    return date in jp_holidays

# 2営業日後の日付を計算する関数
def add_business_days(start_date, num_days, trading_days):
    business_day = start_date
    added_days = 0
    trading_days_set = set(trading_days)  # Set に変換して高速化
    while added_days < num_days:
        business_day += timedelta(days=1)
        # 土日でなく、かつ祝日でない、もしくは trading_days に含まれている場合
        if business_day.weekday() < 5 and (not is_holiday(business_day)) or business_day in trading_days_set:
            added_days += 1
    return business_day

# ロールオーバーの日数を計算する関数
def calculate_rollover_days(open_date, current_date, trading_days):
    try:
        rollover_days = (add_business_days(current_date, 2, trading_days) - add_business_days(open_date, 2, trading_days)).days
    except IndexError as e:
        print(f"Error in calculating rollover days: {e}")
        return 0
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
    
    rows = swap_table.find_all('tr')
    
    headers = [clean_text(th.get_text(strip=True)) for th in rows[0].find_all('th')]

    swap_points = []
    for row in rows[1:]:
        cells = row.find_all(['th', 'td'])
        if len(cells) != len(headers):
            continue
        swap_point = {headers[i]: clean_text(cells[i].get_text(strip=True)) for i in range(len(headers))}
        swap_points.append(swap_point)
    
    return swap_points

def rename_swap_points(swap_points):
    for item in swap_points:
        item['通貨名'] = convert_currency_name(item['通貨名'])
    return swap_points

class SwapCalculator:
    def __init__(self, swap_points, pair):
        self.swap_points_dict = self._create_swap_dict(swap_points)
        self.per_order_size = 10000 if pair in ['ZARJPY=X', 'MXNJPY=X'] else 1000

    def _create_swap_dict(self, swap_points):
        swap_dict = {}
        for point in swap_points:
            pair = point.get('通貨名')
            buy_swap = self._safe_float(point.get('買スワップ', 0))
            sell_swap = self._safe_float(point.get('売スワップ', 0))
            swap_dict[pair] = {'buy': buy_swap, 'sell': sell_swap}
        return swap_dict

    def _safe_float(self, value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def get_total_swap_points(self, pair, position, open_date, current_date, order_size, trading_days):
        rollover_days = calculate_rollover_days(open_date, current_date, trading_days)
        if pair not in self.swap_points_dict:
            return 0.0

        swap_value = self.swap_points_dict[pair].get('buy' if "Buy" in position else 'sell', 0)
        return swap_value * rollover_days * order_size / self.per_order_size

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
    return currency_name

if __name__ == "__main__":
    order_size = 1000
    url = 'https://fx.minkabu.jp/hikaku/moneysquare/spreadswap.html'
    html = get_html(url)
    swap_points = parse_swap_points(html)
    swap_points = rename_swap_points(swap_points)
    calculator = SwapCalculator(swap_points, 'USDJPY=X')
    
    total_swap_points = calculator.get_total_swap_points('USDJPY=X', "Buy", datetime(2024, 5, 31), datetime(2024, 6, 12), order_size, [])
    print(total_swap_points)

    a = calculator.get_total_swap_points('USDJPY=X', "Buy", datetime(2024, 6, 4), datetime(2024, 6, 10), order_size, [])
    b = calculator.get_total_swap_points('USDJPY=X', "Buy", datetime(2024, 6, 11), datetime(2024, 6, 11), order_size, [])
    c = calculator.get_total_swap_points('USDJPY=X', "Buy", datetime(2024, 6, 12), datetime(2024, 6, 12), order_size, [])
    print(f"a, b, a+b+c: {a}, {b}, {c}, {a + b + c}")
