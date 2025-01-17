import ccxt
import pandas as pd
from datetime import datetime, timedelta

# Binance Futures 인스턴스 생성 (API Key 없이)
binance = ccxt.binance({
    'options': {'defaultType': 'future'},  # Futures 설정
})

def fetch_binance_futures_ohlcv(symbol, timeframe='1m', start_date='2021-01-01'):
    """
    Binance Futures에서 1분 간격의 시세 데이터를 반복적으로 요청하여 수집합니다.

    Args:
        symbol (str): 거래 쌍 (예: 'BTC/USDT').
        timeframe (str): 데이터 간격 (기본값: '1m').
        start_date (str): 시작 날짜 (YYYY-MM-DD 형식).

    Returns:
        pd.DataFrame: OHLCV 데이터를 포함한 DataFrame.
    """
    all_data = []  # 데이터를 저장할 리스트
    since = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)  # 시작 시간 (밀리초)

    while True:
        try:
            # Binance Futures에서 OHLCV 데이터 요청
            ohlcv = binance.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1500)

            if not ohlcv:  # 더 이상 데이터가 없으면 종료
                print("더 이상 가져올 데이터가 없습니다.")
                break

            # 데이터를 리스트에 추가
            all_data.extend(ohlcv)

            # 마지막 타임스탬프 이후로 데이터 요청 시작 시간 이동
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 60000  # 1분 추가

            # 종료 조건: 현재 시간 초과 시 종료
            if last_timestamp >= int(datetime.utcnow().timestamp() * 1000):
                print("현재 시간에 도달했습니다. 데이터 수집을 종료합니다.")
                break

            print(f"현재 수집된 데이터 포인트 수: {len(all_data)}")
        except Exception as e:
            print(f"오류 발생: {e}")
            break

    # 데이터를 Pandas DataFrame으로 변환
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # 타임스탬프를 날짜/시간 형식으로 변환
    return df

if __name__ == "__main__":
    # 거래 쌍, 데이터 간격 및 시작 날짜 설정
    symbol = "BTC/USDT"
    timeframe = "1m"  # 1분 간격
    start_date = "2021-01-01"

    # 데이터 수집
    print(f"{symbol} 거래쌍의 {timeframe} 간격 데이터를 수집합니다...")
    data = fetch_binance_futures_ohlcv(symbol, timeframe, start_date)

    # 결과 출력
    print(f"총 데이터 포인트 수: {len(data)}")  # 수집된 데이터 포인트 개수 출력

    # CSV 파일로 저장
    data.to_csv("btc_futures_data.csv", index=False)
    print("데이터가 btc_futures_data.csv에 저장되었습니다.")
