import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model  # type: ignore
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from tqdm import tqdm  # 진행률 표시를 위한 tqdm

# 파라미터 설정
LOOK_BACK = 60  # 과거 데이터 창 크기
FINAL_MODEL_PATH = "models/final_decision_model.h5"  # 최종 모델 경로

# 데이터 전처리 함수
def preprocess_data(data, look_back, scaler):
    """
    데이터를 전처리하여 예측에 사용할 입력 데이터를 생성합니다.

    Args:
        data (list): 예측에 사용할 종가 데이터.
        look_back (int): 과거 데이터 창 크기.
        scaler (MinMaxScaler): 스케일러 객체.

    Returns:
        np.array: 모델 입력 데이터.
    """
    if len(data) < look_back:
        raise ValueError("데이터 길이가 LOOK_BACK보다 작습니다. 충분한 초기 데이터를 제공하세요.")

    scaled_data = scaler.transform(np.array(data).reshape(-1, 1))  # 스케일링 수행
    x_data = [scaled_data[i - look_back:i, 0] for i in range(look_back, len(scaled_data) + 1)]
    return np.array(x_data)

# 시각화 함수
def visualize_predictions(timestamps, predicted_prices):
    """
    예측된 시세 데이터를 시각화합니다.

    Args:
        timestamps (list): 시간 데이터.
        predicted_prices (list): 예측된 시세 데이터.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(timestamps, predicted_prices, label="Predicted Price", color="blue", linewidth=2)
    plt.title("Predicted Prices from 2025-01-15 02:37")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.xticks(ticks=timestamps[::60], labels=[ts.strftime("%Y-%m-%d %H:%M") for ts in timestamps[::60]], rotation=45)
    plt.yticks(np.arange(min(predicted_prices) - 500, max(predicted_prices) + 500, 500))
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # 데이터 로드
    data = pd.read_csv("btc_futures_data.csv")  # 과거 데이터를 가져옴

    # 2025-01-15 02:37:00 이후부터 예측 시작
    start_time = datetime.strptime("2025-01-15 02:37:00", "%Y-%m-%d %H:%M:%S")
    current_time = datetime.utcnow()

    # 1분 간격으로 예측할 타임스탬프 생성
    timestamps = [start_time + timedelta(minutes=i) for i in range(int((current_time - start_time).total_seconds() // 60))]

    # 초기 데이터 준비
    initial_data = data["close"].iloc[-LOOK_BACK:].tolist()  # 가장 최근의 LOOK_BACK만큼의 데이터 가져옴
    if len(initial_data) < LOOK_BACK:
        raise ValueError("LOOK_BACK 길이를 충족하는 초기 데이터가 부족합니다.")

    # 스케일러 생성 및 초기화
    scaler = MinMaxScaler()
    scaler.fit(np.array(initial_data).reshape(-1, 1))  # 초기 데이터를 기준으로 스케일러 피팅

    # 학습된 모델 로드
    final_model = load_model(FINAL_MODEL_PATH)

    # 예측 반복
    predicted_prices = []
    current_data = initial_data.copy()

    print("Starting prediction...")
    for idx in tqdm(range(len(timestamps)), desc="Predicting"):
        # 입력 데이터 전처리
        try:
            x_input = preprocess_data(current_data, LOOK_BACK, scaler)
            x_input = np.expand_dims(x_input[-1], axis=0).reshape(1, LOOK_BACK, 1)  # 모델 입력 형태로 변환

            # 예측 수행
            predicted_price_scaled = final_model.predict(x_input, verbose=0)  # verbose=0으로 출력 제거
            predicted_price = scaler.inverse_transform([[predicted_price_scaled[0][0]]])[0, 0]  # 예측 결과 복원

            # 디버깅 출력
            if idx < 5:  # 처음 5개 예측 결과만 출력
                print(f"\nStep {idx + 1}: Scaled Prediction = {predicted_price_scaled}, Actual Prediction = {predicted_price}")

            # 결과 저장
            predicted_prices.append(predicted_price)

            # 다음 예측을 위한 데이터 업데이트
            current_data.append(predicted_price)
            current_data.pop(0)  # 오래된 데이터 제거
        except Exception as e:
            print(f"Error at step {idx}: {e}")
            break

    # 결과 시각화
    if predicted_prices:
        visualize_predictions(timestamps[:len(predicted_prices)], predicted_prices)
    else:
        print("No predictions to visualize.")