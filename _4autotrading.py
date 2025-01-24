import ccxt
import pandas as pd
import time
import os
from datetime import datetime, timedelta
import threading
from dotenv import load_dotenv

dotenv_path = 'Main/BinanceFutures.env'
load_dotenv(dotenv_path=dotenv_path)

api_key = os.getenv('BINANCE_API_KEY')
secret_key = os.getenv('BINANCE_SECRET_KEY')

print(api_key)
print(secret_key)

# Binance Futures API 설정
binance = ccxt.binance({
    'apiKey': api_key,
    'secret': secret_key,
    'options': {'defaultType': 'future'},  # Futures 데이터 활성화
})

# CSV 파일 경로
csv_file = "./Main/btc_futures_data.csv"
symbol = "BTC/USDT"  # 거래 쌍
timeframe = "1m"  # 1분 간격
limit = 1440

def get_last_timestamp(file_path):
    """
    CSV 파일에서 마지막 타임스탬프를 읽어옵니다.
    """
    if not os.path.exists(file_path):
        return None  # 파일이 없으면 None 반환
    data = pd.read_csv(file_path)
    if data.empty:
        return None  # 파일이 비어 있으면 None 반환
    return pd.to_datetime(data['timestamp'].iloc[-1])  # 마지막 타임스탬프 반환


def fetch_new_data(symbol, timeframe, since):
    """
    Binance에서 새로운 데이터를 가져옵니다.
    """
    all_data = []
    while True:
        try:
            # 데이터를 요청 (since 값은 밀리초로 변환)
            ohlcv = binance.fetch_ohlcv(symbol, timeframe, since=int(since.timestamp() * 1000), limit=limit)
            if not ohlcv:
                print("더 이상 가져올 데이터가 없습니다.")
                break

            # 데이터를 리스트에 추가
            all_data.extend(ohlcv)

            # 마지막 타임스탬프를 갱신
            since = datetime.utcfromtimestamp(ohlcv[-1][0] / 1000) + timedelta(minutes=1)

            # 현재 시간을 초과하면 중단
            if since >= datetime.utcnow():
                break

            print(f"현재까지 {len(all_data)}개의 데이터를 가져왔습니다.")
            time.sleep(1)  # API 제한을 피하기 위해 대기
        except Exception as e:
            print(f"데이터 가져오기 오류: {e}")
            break

    # 데이터를 DataFrame으로 변환
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = pd.DataFrame(all_data, columns=columns)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # 타임스탬프를 변환
    return df


def update_csv(file_path, new_data):
    """
    새로운 데이터를 CSV 파일에 추가합니다.
    """
    if new_data.empty:
        print("추가할 새로운 데이터가 없습니다.")
        return

    # 기존 데이터 불러오기
    if os.path.exists(file_path):
        existing_data = pd.read_csv(file_path)
    else:
        existing_data = pd.DataFrame()

    # 기존 데이터와 병합 및 중복 제거
    updated_data = pd.concat([existing_data, new_data]).drop_duplicates(subset="timestamp").reset_index(drop=True)
    updated_data.to_csv(file_path, index=False)
    print(f"{len(new_data)}개의 새로운 데이터가 추가되었습니다.")


def backfill_to_current_time(file_path, symbol, timeframe):
    """
    CSV 파일의 마지막 타임스탬프부터 현재 시간까지 데이터를 가져옵니다.
    """
    last_timestamp = get_last_timestamp(file_path)
    if last_timestamp is None:
        print("CSV 파일이 없거나 비어 있습니다. 데이터를 2022-01-01부터 가져옵니다.")
        last_timestamp = datetime.strptime("2022-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")

    # 현재 시간까지 데이터를 가져오기
    current_time = datetime.utcnow()
    while last_timestamp < current_time - timedelta(minutes=1):
        print(f"데이터 요청: {last_timestamp}부터 {symbol} 데이터를 가져옵니다.")
        new_data = fetch_new_data(symbol, timeframe, since=last_timestamp + timedelta(minutes=1))

        # 새로운 데이터가 없으면 종료
        if new_data.empty:
            print("더 이상 가져올 데이터가 없습니다.")
            break

        # CSV 파일 업데이트
        update_csv(file_path, new_data)

        # 마지막 타임스탬프 갱신
        last_timestamp = pd.to_datetime(new_data['timestamp'].iloc[-1])

    print("과거 데이터를 현재 시간까지 모두 업데이트했습니다.")


def collect_real_time_data(file_path, symbol, timeframe):
    """
    실시간 데이터를 1분 간격으로 가져옵니다.
    """
    while True:
        try:
            last_timestamp = get_last_timestamp(file_path)
            if last_timestamp is None:
                print("CSV 파일이 없거나 비어 있습니다. 백필 데이터를 먼저 수행합니다.")
                backfill_to_current_time(file_path, symbol, timeframe)
                continue

            print(f"실시간 데이터 요청: {last_timestamp} 이후 데이터를 가져옵니다.")
            new_data = fetch_new_data(symbol, timeframe, since=last_timestamp + timedelta(minutes=1))
            if not new_data.empty:
                update_csv(file_path, new_data)

        except Exception as e:
            print(f"실시간 데이터 업데이트 중 오류 발생: {e}")

        # 1분 대기
        print("1분 대기 중...")
        time.sleep(60)


def main():
    """
    프로그램의 메인 함수: 과거 데이터 백필 및 실시간 업데이트 실행
    """
    print("===== 데이터 수집 프로그램 시작 =====")
    backfill_to_current_time(csv_file, symbol, timeframe)
    collect_real_time_data(csv_file, symbol, timeframe)


if __name__ == "__main__":
    main()