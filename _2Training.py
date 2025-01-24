import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical

# 파라미터 설정
LOOK_BACK = 60  # 과거 데이터 창 크기
LEVERAGE = 10  # 레버리지
PROFIT_TARGET = 0.5  # 목표 수익 (예: 0.5%)

# 익절 가격 계산 함수
# 롱 포지션
def calculate_long_exit_price(entry_price, profit, leverage):
    return entry_price * (100 + (profit + 0.08 * leverage) / leverage) / 100

# 숏 포지션
def calculate_short_exit_price(entry_price, profit, leverage):
    return entry_price * (100 - (profit + 0.08 * leverage) / leverage) / 100

# 데이터 생성 함수
# CSV 파일에서 데이터를 불러오거나 시뮬레이션에 사용할 데이터를 생성

def preprocess_data(file_path, start_time):
    """
    초기 데이터를 읽고 시뮬레이션에 필요한 형태로 변환합니다.
    """
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[df['timestamp'] >= start_time]
    return df

# LSTM 모델 생성 함수
def create_model():
    """LSTM 모델 생성"""
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(LOOK_BACK, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=3, activation='softmax'))  # 매수, 매도, 유지
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 시뮬레이션 함수
# 매수, 매도, 유지를 학습시키며 익절 가격을 기준으로 진입 가격을 업데이트

def simulate_trading(data, model, position_type):
    """
    데이터 없이 매수, 매도, 유지의 최적 행동을 학습시키는 시뮬레이션.
    """
    entry_price = 50  # 초기 진입가
    x_train, y_train = [], []

    for i in range(len(data) - 1):
        # 익절 가격 계산
        if position_type == "long":
            exit_price = calculate_long_exit_price(entry_price, PROFIT_TARGET, LEVERAGE)
        elif position_type == "short":
            exit_price = calculate_short_exit_price(entry_price, PROFIT_TARGET, LEVERAGE)

        # 현재 가격과 익절 가격 비교
        current_price = data['close'].iloc[i]
        next_price = data['close'].iloc[i + 1]

        # 매수, 매도, 유지 결정
        if position_type == "long" and next_price >= exit_price:
            action = 0  # 매수
            entry_price = exit_price  # 진입가 업데이트
        elif position_type == "short" and next_price <= exit_price:
            action = 1  # 매도
            entry_price = exit_price  # 진입가 업데이트
        else:
            action = 2  # 유지

        # 학습 데이터 생성
        x_train.append(data['close'].iloc[max(0, i - LOOK_BACK):i].values)
        y_train.append(action)

    # 데이터 전처리 및 모델 학습
    x_train = np.expand_dims(np.array(x_train), axis=2)
    y_train = to_categorical(np.array(y_train), num_classes=3)

    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)

    return model

if __name__ == "__main__":
    # 초기 데이터 로드
    file_path = "NewFile/btc_futures_data.csv"
    start_time = pd.Timestamp("2022-01-01 00:00:00")
    data = preprocess_data(file_path, start_time)

    # 롱 포지션 모델 학습
    print("롱 포지션 모델 학습 시작...")
    long_model = create_model()
    long_model = simulate_trading(data, long_model, position_type="long")
    long_model.save("long_position_model.h5")

    # 숏 포지션 모델 학습
    print("숏 포지션 모델 학습 시작...")
    short_model = create_model()
    short_model = simulate_trading(data, short_model, position_type="short")
    short_model.save("short_position_model.h5")

    print("모든 모델 학습 완료!")
