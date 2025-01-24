import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler

# 파라미터 설정
LOOK_BACK = 60  # 과거 데이터 창 크기
THRESHOLD = 0.01  # 가격 변화 기준
PROFIT_TARGET = 0.5  # 목표 수익률 (%)
LONG_MODEL_PATH = "models/long_position_model.h5"  # 롱 모델 경로
SHORT_MODEL_PATH = "models/short_position_model.h5"  # 숏 모델 경로
FINAL_MODEL_PATH = "models/final_decision_model.h5"  # 최종 모델 저장 경로

# 데이터 전처리 함수
def preprocess_data(data, look_back):
    """
    데이터를 전처리하여 모델 입력 형식으로 변환합니다.
    """
    close_prices = data['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)

    x_data = []
    for i in range(look_back, len(scaled_data)):
        x_data.append(scaled_data[i-look_back:i, 0])

    return np.array(x_data), scaler

# 롱/숏 모델 로드
def load_models():
    long_model = load_model(LONG_MODEL_PATH)
    short_model = load_model(SHORT_MODEL_PATH)
    return long_model, short_model

# 익절 가격 계산 함수
def calculate_take_profit(entry_price, profit_target, leverage, position):
    """
    익절 가격을 계산합니다.
    롱 포지션: 익절 가격 = 진입가 * (100 + (수익 + 0.08 * 레버리지) / 레버리지) / 100
    숏 포지션: 익절 가격 = 진입가 * (100 - (수익 + 0.08 * 레버리지) / 레버리지) / 100
    """
    if position == "long":
        return entry_price * (100 + (profit_target + 0.08 * leverage) / leverage) / 100
    elif position == "short":
        return entry_price * (100 - (profit_target + 0.08 * leverage) / leverage) / 100

# 최종 행동 데이터 생성
def generate_final_training_data(data, long_model, short_model, look_back, profit_target, leverage):
    """
    롱 모델과 숏 모델의 판단을 기반으로 최선의 행동을 학습할 데이터를 생성합니다.
    """
    x_data, scaler = preprocess_data(data, look_back)
    x_data = np.expand_dims(x_data, axis=2)

    # 롱 모델과 숏 모델의 예측 수행
    long_predictions = long_model.predict(x_data, verbose=0)
    short_predictions = short_model.predict(x_data, verbose=0)

    entry_prices = data['close'].iloc[look_back:].values
    x_train, y_train = [], []

    for i, (long_pred, short_pred, entry_price) in enumerate(zip(long_predictions, short_predictions, entry_prices)):
        # 각 포지션의 익절 가격 계산
        long_take_profit = calculate_take_profit(entry_price, profit_target, leverage=10, position="long")
        short_take_profit = calculate_take_profit(entry_price, profit_target, leverage=10, position="short")

        # 행동 점수 계산
        long_score = long_pred[0] - abs(entry_price - long_take_profit)
        short_score = short_pred[0] - abs(entry_price - short_take_profit)

        # 최종 행동 결정
        if long_score > short_score:
            if long_pred[0] > max(long_pred[1], long_pred[2]):
                y_train.append(0)  # 롱 매수
            elif long_pred[1] > max(long_pred[0], long_pred[2]):
                y_train.append(1)  # 롱 매도
            else:
                y_train.append(2)  # 롱 유지
        else:
            if short_pred[0] > max(short_pred[1], short_pred[2]):
                y_train.append(0)  # 숏 매수
            elif short_pred[1] > max(short_pred[0], short_pred[2]):
                y_train.append(1)  # 숏 매도
            else:
                y_train.append(2)  # 숏 유지

        x_train.append(x_data[i])

    return np.array(x_train), np.array(y_train), scaler

# 최종 모델 생성
def create_final_model(input_shape):
    """
    최종 행동 모델을 생성합니다.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=3, activation="softmax"))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# 학습 실행
if __name__ == "__main__":
    # CSV 데이터 로드
    data = pd.read_csv("btc_futures_data.csv")
    
    # 모델 로드
    long_model, short_model = load_models()

    # 최종 학습 데이터 생성
    x_train, y_train, scaler = generate_final_training_data(data, long_model, short_model, LOOK_BACK, PROFIT_TARGET, leverage=10)

    # 최종 모델 생성 및 학습
    final_model = create_final_model(input_shape=(LOOK_BACK, 1))
    history = final_model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)

    # 학습된 모델 저장
    final_model.save(FINAL_MODEL_PATH)
    print(f"최종 모델이 {FINAL_MODEL_PATH}에 저장되었습니다.")
