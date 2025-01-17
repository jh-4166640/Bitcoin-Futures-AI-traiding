import numpy as np
import ccxt
import random

class QLearningBitcoinTrader:
    def __init__(self, api_key, api_secret, symbol="BTC/USDT", balance=10000, alpha=0.1, gamma=0.9, epsilon=0.1):
        # 거래소와 연결 (Binance 예시)
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
        })
        
        self.symbol = symbol
        self.balance = balance  # 초기 자산
        self.position = 0  # 현재 포지션 (1: 롱, -1: 숏, 0: 없음)
        self.entry_price = None  # 포지션 진입 가격
        self.alpha = alpha  # 학습률
        self.gamma = gamma  # 할인율
        self.epsilon = epsilon  # 탐험율 (exploration rate)

        # Q-테이블 초기화
        self.q_table = {}  # Q-value를 저장할 테이블
        self.actions = [1, 2, 3, 4, 5]  # 롱 진입, 롱 청산, 숏 진입, 숏 청산, 유지

    def get_balance(self):
        balance = self.exchange.fetch_balance()
        return balance['total']['USDT']

    def get_current_price(self):
        ticker = self.exchange.fetch_ticker(self.symbol)
        return ticker['last']

    def get_state(self):
        # 상태는 간단히 비트코인 가격과 현재 포지션을 합친 값을 사용할 수 있습니다.
        current_price = self.get_current_price()
        state = (current_price, self.position)  # 상태 정의
        return state

    def get_q_value(self, state, action):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))  # Q-table 초기화
        return self.q_table[state][action]

    def choose_action(self, state):
        # 탐험(ε-greedy) 방식으로 행동 선택
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # 무작위 선택
        else:
            q_values = self.q_table[state]
            return np.argmax(q_values)  # Q-value가 최대인 행동 선택

    def update_q_value(self, state, action, reward, next_state):
        # Q-value 업데이트
        max_future_q = np.max(self.q_table[next_state]) if next_state in self.q_table else 0
        current_q = self.q_table[state][action]
        # Q-learning 업데이트 공식
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state][action] = new_q

    def enter_long(self, amount):
        # 롱 포지션 진입
        self.exchange.create_market_buy_order(self.symbol, amount)
        self.position = 1
        self.entry_price = self.get_current_price()

    def exit_long(self, amount):
        # 롱 포지션 청산
        self.exchange.create_market_sell_order(self.symbol, amount)
        self.position = 0

    def enter_short(self, amount):
        # 숏 포지션 진입
        self.exchange.create_market_sell_order(self.symbol, amount)
        self.position = -1
        self.entry_price = self.get_current_price()

    def exit_short(self, amount):
        # 숏 포지션 청산
        self.exchange.create_market_buy_order(self.symbol, amount)
        self.position = 0

    def hold(self):
        # 포지션 유지
        pass

    def reward(self):
        # 간단한 보상 함수 예시: 포지션 청산 시 이익/손실 계산
        if self.position == 1:  # 롱 포지션
            return self.get_current_price() - self.entry_price
        elif self.position == -1:  # 숏 포지션
            return self.entry_price - self.get_current_price()
        else:
            return 0  # 포지션이 없으면 보상은 0

    def trade(self):
        state = self.get_state()
        action = self.choose_action(state)
        
        amount = 0.01  # 예시로 0.01 BTC를 거래하는 것으로 설정

        # 행동에 따른 매매
        if action == 1:  # 롱 포지션 진입
            self.enter_long(amount)
        elif action == 2:  # 롱 포지션 청산
            self.exit_long(amount)
        elif action == 3:  # 숏 포지션 진입
            self.enter_short(amount)
        elif action == 4:  # 숏 포지션 청산
            self.exit_short(amount)
        elif action == 5:  # 포지션 유지
            self.hold()

        reward_value = self.reward()  # 보상 계산
        next_state = self.get_state()

        # Q-value 업데이트
        self.update_q_value(state, action, reward_value, next_state)

        return reward_value

if __name__ == "__main__":
    api_key = "your_api_key"
    api_secret = "your_api_secret"
    
    # Q-Learning 트레이딩 시스템 초기화
    trader = QLearningBitcoinTrader(api_key, api_secret)
    
    # 100번의 트레이딩 에피소드 실행
    for episode in range(100):
        total_reward = 0
        for step in range(100):  # 각 에피소드에서 100번의 트레이딩을 진행
            reward = trader.trade()
            total_reward += reward
            print(f"Episode: {episode + 1}, Step: {step + 1}, Total Reward: {total_reward}")
        print(f"Episode {episode + 1} finished with total reward: {total_reward}")
