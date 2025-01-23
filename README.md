\- bitCoin 거래 API(무료, 실제 시간보다 9시간 이전)를 불러오기 위해서는 apiFetch.py를 이용해 시간 t{t>1분} 단위로 Open, High, Low, Close, Volume 등 가격과 거래량 데이터를 불러온다.
*BTCUSDTyyyymmdd.csv

\- 불러온 데이터는 LSTM_BTCUSDT 파일에서 tensorflow의 keras LSTM으로 미래의 가격을 예측할 수 있도록 학습시킨다.

\- TrainingEnv.py 파일에서 가상 트레이딩 환경을 구축하여 Q-learning으로 현재 가격에 대해 Position 선택(Long, Short, Long Exit, Short Exit, Hold)하는 모델을 만들어낸다.

# 추후 구현해야 할 것
1) LSTM모델을 통해 예측한 데이터를 Q-learning 모델과 합쳐 현재 더 나은 판단을 할 수 있게 한다.
2) Binance(암호화폐 거래소)의 Test Net에서 배포 전 모델 검증을 실시하여 수익률을 집계할 수 있게 한다.
3) 실시간 API에 접속, 실제 거래 환경과 연동하여 실시간 실제 선물 거래를 진행한다.
4) 유지/보수 및 주기적인 예측 모델, 거래 모델 업데이트
5) ++ 상품화 과정 : 자동 업데이트 되도록 실시간 데이터를 쌓는다.

