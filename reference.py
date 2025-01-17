import gym
from gym import spaces
import numpy as np

class BitcoinTradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000, fee_rate=0.001):
        super(BitcoinTradingEnv, self).__init__()
        
        # 트레이딩 데이터를 로드
        self.data = data
        self.current_step = 0
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0  # 현재 포지션 (롱: +1, 숏: -1, 없음: 0)
        self.net_worth = initial_balance
        self.max_steps = len(data) - 1
        self.fee_rate = fee_rate  # 거래 수수료율 (예: 0.1% = 0.001)

        # Action space: 0 - Hold, 1 - Long Entry, 2 - Long Exit, 3 - Short Entry, 4 - Short Exit
        self.action_space = spaces.Discrete(5)

        # Observation space: 가격, 잔액, 포지션 상태
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(len(data.columns) + 2,), dtype=np.float32
        )

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.net_worth = self.initial_balance
        return self._next_observation()

    def _next_observation(self):
        obs = np.append(
            self.data.iloc[self.current_step].values,
            [self.balance, self.position]
        )
        return obs

    def step(self, action):
        # 현재 가격 정보
        current_price = self.data.iloc[self.current_step]["close"]
        reward = 0

        # 행동 처리
        if action == 1:  # Long Entry
            if self.position == 0:  # 새로운 롱 포지션 진입
                trade_cost = self.balance * self.fee_rate  # 수수료 계산
                if self.balance > trade_cost:  # 잔액이 충분한 경우
                    self.balance -= trade_cost
                    self.position = self.balance / current_price  # 포지션 크기
                    self.balance = 0
        elif action == 2:  # Long Exit
            if self.position > 0:  # 롱 포지션 종료
                trade_value = self.position * current_price
                trade_cost = trade_value * self.fee_rate  # 수수료 계산
                self.balance = trade_value - trade_cost
                self.position = 0
        elif action == 3:  # Short Entry
            if self.position == 0:  # 새로운 숏 포지션 진입
                trade_cost = self.balance * self.fee_rate  # 수수료 계산
                if self.balance > trade_cost:  # 잔액이 충분한 경우
                    self.balance -= trade_cost
                    self.position = -self.balance / current_price  # 숏 포지션 크기
                    self.balance = 0
        elif action == 4:  # Short Exit
            if self.position < 0:  # 숏 포지션 종료
                trade_value = abs(self.position) * current_price
                trade_cost = trade_value * self.fee_rate  # 수수료 계산
                self.balance = trade_value - trade_cost
                self.position = 0

        # 잔액 업데이트
        self.net_worth = self.balance + (self.position * current_price)

        # 스텝 업데이트
        self.current_step += 1
        done = self.current_step >= self.max_steps

        # 보상 계산 (순자산 증가/감소)
        reward = self.net_worth - self.initial_balance

        return self._next_observation(), reward, done, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}")
        print(f"Net Worth: {self.net_worth}")
        print(f"Balance: {self.balance}")
        print(f"Position: {self.position}")
