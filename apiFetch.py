import ccxt
import pandas as pd
from datetime import datetime

def fetch_historical_data(symbol, timeframe, since, limit=1500):
    exchange = ccxt.binanceusdm()
    all_data = []
    
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=limit)
        if not ohlcv:
            break
        all_data.extend(ohlcv)
        since = ohlcv[-1][0] + 1  # 마지막 데이터 이후부터 가져오기
    return all_data



# Binance Futures 객체 생성
exchange = ccxt.binanceusdm()  # USDT-M 선물 마켓

# 과거 데이터 가져오기
symbol = "BTC/USDT"
timeframe = "1m"
since = exchange.parse8601("2022-01-02T00:00:00Z")  # 시작 날짜
data = fetch_historical_data(symbol, timeframe, since)

# DataFrame 변환 및 저장
columns = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
df = pd.DataFrame(data, columns=columns)
df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
df.to_csv(f"{symbol.replace('/', '_')}_historical_data.csv", index=False)
print(f"CSV 저장 완료!")
