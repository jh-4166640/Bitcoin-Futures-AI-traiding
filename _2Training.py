import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential             
from tensorflow.keras.layers import Dense, Dropout, LSTM   
from tensorflow.keras.utils import to_categorical          

# 파라미터 설정
LOOK_BACK = 60  # 과거 데이터 창 크기
THRESHOLD = 0.01  # 가격 변화 기준
EPOCHS = 10  # 에포크 수
BATCH_SIZE = 32  # 배치 크기
LONG_MODEL_PATH = "Main/models/long_position_model.h5"  # 롱 모델 저장 경로
SHORT_MODEL_PATH = "Main/models/short_position_model.h5"  # 숏 모델 저장 경로


# 데이터 전처리 함수
def preprocess_data(data, look_back, threshold, position="long"):
    """
    데이터를 전처리하고 롱/숏 포지션에 따른 매수/매도/유지 신호를 생성합니다.

    Args:
        data (pd.DataFrame): OHLCV 데이터.
        look_back (int): 과거 데이터 창 크기.
        threshold (float): 가격 변화율 기준.
        position (str): "long" 또는 "short" 포지션을 지정.

    Returns:
        x_data, y_data, scaler: 입력 데이터, 신호 데이터, MinMaxScaler 객체.
    """
    close_prices = data["close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)

    x_data, y_data = [], []
    for i in range(look_back, len(scaled_data) - 1):
        # 과거 데이터 창 생성
        x_data.append(scaled_data[i - look_back:i, 0])

        # 현재와 다음 가격 변화율 계산
        price_now = scaled_data[i, 0]
        price_next = scaled_data[i + 1, 0]
        price_change = (price_next - price_now) / price_now

        # 롱/숏 포지션에 따른 신호 생성
        if position == "long":
            if price_change > threshold:
                y_data.append(0)  # 매수
            elif price_change < -threshold:
                y_data.append(1)  # 매도
            else:
                y_data.append(2)  # 유지
        elif position == "short":
            if price_change > threshold:
                y_data.append(1)  # 매도
            elif price_change < -threshold:
                y_data.append(0)  # 매수
            else:
                y_data.append(2)  # 유지

    return np.array(x_data), np.array(y_data), scaler

# LSTM 모델 생성 함수
def create_model():
    """
    LSTM 모델을 생성합니다.

    Returns:
        Sequential: 생성된 LSTM 모델.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(LOOK_BACK, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=3, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# 모델 학습 및 시뮬레이션 함수
def simulate_and_train(data, model_path, position, look_back=60, epochs=10, batch_size=32):
    """
    롱/숏 포지션에 따라 매수/매도/유지 신호를 학습하고 시뮬레이션을 수행합니다.

    Args:
        data (pd.DataFrame): 학습 및 시뮬레이션 데이터.
        model_path (str): 모델 저장 경로.
        position (str): "long" 또는 "short".
        look_back (int): 과거 데이터 창 크기.
        epochs (int): 학습 에포크 수.
        batch_size (int): 배치 크기.
    """
    x_data, y_data, scaler = preprocess_data(data, look_back, THRESHOLD, position)

    # 데이터 차원 확장 및 신호를 원-핫 인코딩
    x_data = np.expand_dims(x_data, axis=2)
    y_data = to_categorical(y_data, num_classes=3)

    # 모델 생성
    model = create_model()

    # 학습 루프
    for epoch in range(epochs):
        history = model.fit(x_data, y_data, batch_size=batch_size, epochs=1, verbose=1)
        print(f"[{position.upper()}] Epoch {epoch+1}/{epochs}, Loss: {history.history['loss'][0]:.4f}")

    # 학습된 모델 저장
    model.save(model_path)
    print(f"{position.capitalize()} model saved to {model_path}.")


if __name__ == "__main__":
    # 데이터 로드
    data = pd.read_csv("Main/btc_futures_data.csv")

    # 롱 포지션 학습 및 모델 저장
    simulate_and_train(data, LONG_MODEL_PATH, position="long", look_back=LOOK_BACK, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # 숏 포지션 학습 및 모델 저장
    simulate_and_train(data, SHORT_MODEL_PATH, position="short", look_back=LOOK_BACK, epochs=EPOCHS, batch_size=BATCH_SIZE)
