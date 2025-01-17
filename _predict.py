import numpy as np
import pandas as pd
from datetime import datetime
import sys
from tensorflow.keras.models import load_model              #type: ignore
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 설정된 파일 경로 및 파라미터
FINAL_MODEL_PATH = "models/final_decision_model.h5"  # 최종 모델 경로
LOOK_BACK = 60  # 과거 데이터 창 크기
THRESHOLD = 0.01  # 가격 변화 기준

# 데이터 불러오기
data = pd.read_csv("btc_futures_data.csv")
data['timestamp'] = pd.to_datetime(data['timestamp'])  # 타임스탬프 변환

# 현재 시간을 기준으로 예측할 데이터 생성
current_time = datetime.utcnow()
last_data_time = data['timestamp'].iloc[-1]

def generate_future_data(data, start_time, end_time):
    """
    2025-01-15 02:38부터 현재 시간까지 예측하기 위한 데이터를 생성합니다.

    Args:
        data (pd.DataFrame): 기존 OHLCV 데이터.
        start_time (datetime): 예측 시작 시간.
        end_time (datetime): 예측 종료 시간.

    Returns:
        pd.DataFrame: 추가된 타임스탬프와 종가 데이터를 포함한 데이터프레임.
    """
    timestamps = pd.date_range(start=start_time, end=end_time, freq='1min')
    additional_data = pd.DataFrame(timestamps, columns=['timestamp'])

    # 빈 종가를 기존 마지막 종가로 채움
    additional_data['close'] = data['close'].iloc[-1]
    return additional_data

# 추가 데이터 생성
future_data = generate_future_data(data, last_data_time, current_time)
data = pd.concat([data, future_data]).reset_index(drop=True)

# 데이터 전처리
scaler = MinMaxScaler()
data['scaled_close'] = scaler.fit_transform(data['close'].values.reshape(-1, 1))

def predict_future_prices(data, model, look_back):
    """
    모델을 사용하여 미래 시세를 예측합니다.

    Args:
        data (pd.DataFrame): 전처리된 데이터.
        model (Model): LSTM 모델.
        look_back (int): 과거 데이터 창 크기.

    Returns:
        np.array: 예측된 종가 데이터.
    """
    scaled_data = data['scaled_close'].values
    predictions = []

    for i in range(look_back, len(scaled_data)):
        input_data = scaled_data[i-look_back:i].reshape(1, look_back, 1)
        prediction = model.predict(input_data, verbose=0)

        # 예측된 클래스(매수, 매도, 유지)에 따라 종가 변화 계산
        if np.argmax(prediction) == 0:  # 매수
            next_price = scaled_data[i] * (1 + THRESHOLD)
        elif np.argmax(prediction) == 1:  # 매도
            next_price = scaled_data[i] * (1 - THRESHOLD)
        else:  # 유지
            next_price = scaled_data[i]

        predictions.append(next_price)

        # 진행 상황 출력 (한 줄)
        sys.stdout.write(f"\r진행 상황: {i}/{len(scaled_data) - look_back} 데이터 포인트 처리 중...")
        sys.stdout.flush()

    return np.array(predictions)

# 최종 모델 로드
final_model = load_model(FINAL_MODEL_PATH)

# 미래 가격 예측
print("미래 시세 예측을 시작합니다...")
predicted_prices = predict_future_prices(data, final_model, LOOK_BACK)
print("미래 시세 예측이 완료되었습니다.")

# 스케일 복원
predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))

# 시각화
plt.figure(figsize=(12, 6))
plt.plot(data['timestamp'].iloc[-len(predicted_prices):], data['close'].iloc[-len(predicted_prices):], label="실제 가격", alpha=0.7)
plt.plot(data['timestamp'].iloc[-len(predicted_prices):], predicted_prices, label="예측 가격", linestyle="--")
plt.xlabel("시간")
plt.ylabel("가격 (USDT)")
plt.title("BTC/USDT 미래 시세 예측")
plt.legend()
plt.grid()
plt.show()
