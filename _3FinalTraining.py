import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 텐서플로우 경고 메시지 최소화
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 파라미터 설정
LOOK_BACK = 60  # 과거 데이터 창 크기
THRESHOLD = 0.01  # 가격 변화 기준
LONG_MODEL_PATH = " Main/models/long_position_model.h5"  # 롱 모델 경로
SHORT_MODEL_PATH = " Main/models/short_position_model.h5"  # 숏 모델 경로
FINAL_MODEL_PATH = " Main/models/final_decision_model.h5"  # 최종 모델 저장 경로

# 데이터 전처리 함수
def preprocess_data(data, look_back):
    """
    데이터를 전처리하여 모델 입력 형식으로 변환합니다.

    Args:
        data (pd.DataFrame): OHLCV 데이터.
        look_back (int): 과거 데이터 창 크기.

    Returns:
        np.array: 입력 데이터.
        MinMaxScaler: 데이터 스케일러 객체.
    """
    if len(data) <= look_back:
        raise ValueError(f"데이터 길이가 {look_back}보다 작습니다. 충분한 데이터를 제공해주세요.")

    close_prices = data["close"].values.reshape(-1, 1)  # 종가 데이터를 2D 배열로 변환
    scaler = MinMaxScaler()  # 데이터를 0과 1 사이로 스케일링하기 위한 객체 생성
    scaled_data = scaler.fit_transform(close_prices)  # 스케일링 수행

    x_data = []
    for i in range(look_back, len(scaled_data)):
        x_data.append(scaled_data[i - look_back:i, 0])  # 과거 데이터 창 생성

    return np.array(x_data), scaler

# 학습 데이터 생성 함수
def generate_training_data(data, long_model, short_model, entry_prices, position_size):
    """
    롱 모델과 숏 모델의 예측 결과를 기반으로 최선의 행동을 학습하기 위한 데이터를 생성합니다.

    Args:
        data (np.array): 입력 데이터.
        long_model (Model): 사전 학습된 롱 모델.
        short_model (Model): 사전 학습된 숏 모델.
        entry_prices (list): 진입 가격 리스트.
        position_size (float): 포지션 크기.

    Returns:
        np.array: 입력 데이터.
        np.array: 출력 레이블 (최선의 행동).
    """
    if len(data) != len(entry_prices):
        raise ValueError("데이터와 진입 가격 리스트의 길이가 일치하지 않습니다.")

    long_predictions = long_model.predict(data, verbose=0)  # 롱 모델 예측 수행
    short_predictions = short_model.predict(data, verbose=0)  # 숏 모델 예측 수행

    x_train = []
    y_train = []

    for i, (long_pred, short_pred) in enumerate(zip(long_predictions, short_predictions)):
        current_price = entry_prices[i]  # 현재 가격 가져오기

        # 행동 점수 계산
        buy_score = long_pred[0] + short_pred[1]  # 롱 매수 + 숏 매도 점수
        sell_score = long_pred[1] + short_pred[0]  # 롱 매도 + 숏 매수 점수
        hold_score = long_pred[2] + short_pred[2]  # 롱 유지 + 숏 유지 점수

        # 최적의 행동 선택
        if buy_score > max(sell_score, hold_score):
            y_train.append(0)  # 매수
        elif sell_score > max(buy_score, hold_score):
            y_train.append(1)  # 매도
        else:
            y_train.append(2)  # 유지

        x_train.append(data[i])  # 입력 데이터 저장

        # 진행 상황 출력
        if i % 100 == 0:
            print(f"진행 상황: {i}/{len(data)} 데이터 포인트 처리 중...")

    return np.array(x_train), np.array(y_train)

# 최종 모델 생성 함수
def create_final_model(input_shape):
    """
    최종 행동 모델을 생성합니다.

    Args:
        input_shape (tuple): 입력 데이터의 형태.

    Returns:
        Sequential: 생성된 LSTM 모델.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))  # LSTM 레이어 추가
    model.add(Dropout(0.2))  # 과적합 방지를 위한 드롭아웃
    model.add(LSTM(units=50, return_sequences=False))  # 또 다른 LSTM 레이어 추가
    model.add(Dropout(0.2))
    model.add(Dense(units=3, activation="softmax"))  # 매수, 매도, 유지 출력
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    # 데이터 로드
    data = pd.read_csv("Main/btc_futures_data.csv")  # 2021-01-01부터 1분 간격 데이터

    # 데이터 전처리
    x_data, scaler = preprocess_data(data, LOOK_BACK)  # 입력 데이터 생성
    x_data = np.expand_dims(x_data, axis=2)  # LSTM 입력 형태로 변환

    # 진입 가격 리스트 생성
    entry_prices = data["close"].iloc[LOOK_BACK:].values

    # 포지션 크기 설정
    position_size = 1.0  # 포지션 크기 (예: 1 BTC)

    # 기존 모델 불러오기
    long_model = load_model(LONG_MODEL_PATH)  # 롱 모델 불러오기
    short_model = load_model(SHORT_MODEL_PATH)  # 숏 모델 불러오기

    # 학습 데이터 생성
    print("학습 데이터 생성을 시작합니다...")
    x_train, y_train = generate_training_data(x_data, long_model, short_model, entry_prices, position_size)
    print("\n학습 데이터 생성이 완료되었습니다.")

    # 최종 모델 생성
    final_model = create_final_model(input_shape=(LOOK_BACK, 1))

    # 학습 (배치 크기 32로 설정하여 1분 단위 학습 수행)
    history = final_model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)

    # 최종 모델 저장
    final_model.save(FINAL_MODEL_PATH)  # 학습된 모델 저장
    print(f"최종 모델이 {FINAL_MODEL_PATH}에 저장되었습니다.")
