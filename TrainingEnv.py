import numpy as np
import pandas as pd
import random
import sys

class TrainingEnv():
    def __init__(self, data, inital_balance=100, fee_rate=0.001, position_rate = 0.4):
        # data road
        self.data = data
        self.current_step = 0 # model have step on Environment
        self.initial_balance = inital_balance
        self.balance = inital_balance # change value when trading step
        self.position = 0 # current position (None : 0, Long : 1, Short : -1)
        self.net_worth = inital_balance # wallet + keep position value
        self.max_step = len(data) - 2
        self.fee_rate = fee_rate # 거래 수수료
        self.position_rate = position_rate # 포지션 크기 비율
        self.entry_price = 0    
        
        self.alpha = 0.05 # learning rate
        self.gamma = 0.4 # 할인율
        self.epsilon = 0.1 # 탐험율 exploration rate
        
        # Q-tabel init
        self.q_table = {}
        self.actions = [-2,-1,0,1,2]
        
    def setHyperParam(self,alpha, gamma, epsilon):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.current_step = 0
        print(f"alpha {self.alpha},,, gamma {self.gamma},,, epsilon {self.epsilon}")
    def getHyperParam(self):
        return self.alpha, self.gamma, self.epsilon
    def get_q_value(self, state, action):  
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))  # Q-table 초기화
        return self.q_table[state][action]
    
    def choose_action(self, state):
         # 상태가 Q-table에 없으면 초기화
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        # 탐험(ε-greedy) 방식으로 행동 선택
        if random.uniform(0,1) < self.epsilon:
            return random.choice(self.actions) # 무작위 선택
        else :
            q_values = self.q_table[state]
            return np.argmax(q_values) # Q-value가 최대인 행동 선택
        
    def update_q_value(self, state, action, reward, next_state):
        # Q-value 업데이트
        max_future_q = np.max(self.q_table[next_state]) if next_state in self.q_table else 0
        current_q = self.q_table[state][action]
        # Q-learning 업데이트 공식
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state][action] = new_q
        
    def Enter_long(self,current_price):
        if self.position == 0:
            # 거래 수수료 계산 (자산*포지션 진입 비율) * 거래 수수료율
            trade_cost = (self.balance*self.position_rate) * self.fee_rate
            # 현재 자산 > 거래 수수료 : 포지션 진입 가능
            if self.balance > trade_cost:
                # 현재 자산 - (현재 자산*포지션 진입 비율) - 거래수수료
                self.position = (self.balance*self.position_rate)/current_price
                self.balance -= self.balance*self.position_rate - trade_cost            
                
    def Exit_long(self, current_price):
        if self.position > 0:
            trade_value = self.position * current_price
            # 수수료 계산
            trade_cost = trade_value * self.fee_rate
            self.balance += (trade_value - trade_cost)
            self.position = 0
            
    def Enter_short(self, current_price):
        if self.position == 0:
            # 거래 수수료 계산 (자산*포지션 진입 비율) * 거래 수수료율
            trade_cost = (self.balance*self.position_rate) * self.fee_rate
            # 현재 자산 > 거래 수수료 : 포지션 진입 가능
            if self.balance > trade_cost:
                # 현재 자산 - (현재 자산*포지션 진입 비율) - 거래수수료
                self.position = -(self.balance*self.position_rate)/current_price
                self.balance -= self.balance*self.position_rate - trade_cost
                self.entry_price = current_price
                
    def Exit_short(self, current_price):
        if self.position < 0:
            trade_value = abs(self.position) * current_price
            # 수수료 계산
            trade_cost = trade_value * self.fee_rate
            self.balance += (trade_value - trade_cost)
            self.position = 0
            self.entry_price = 0
            
    def step(self):
        # now price information
        current_price = self.data.iloc[self.current_step]["Close"]
        state = (current_price, self.position) # 상태 정의
        action = trade.choose_action(state)
        reward = 0
        # 추가 포지션 진입 없음
        # 롱 포지션 진입
        if action == 1:
            self.Enter_long(current_price)
        # 롱 포지션 정리
        elif action == 2:
            self.Exit_long(current_price)
        # 숏 포지션 진입
        elif action == -1:
            self.Enter_short(current_price)
        # 롱 포지션 정리
        elif action == -2:
            self.Exit_short(current_price)
        
        if self.position < 0:
            profit = self.position*current_price-((self.position*self.entry_price)-(self.position*current_price))
        elif self.position >= 0:
            profit = self.position*current_price
            
        # 자산 업데이트
        self.net_worth = self.balance + profit
        
        # 보상 (자산) 증/감
        reward = self.net_worth - self.initial_balance
        #if abs(action) == 2:
        #    print(f"price: {current_price}, acntion: {action}, position: {self.position}")
        self.current_step += 1
        next_price = self.data.iloc[self.current_step]["Close"]
        next_state = (next_price, self.position)
        self.update_q_value(state, action,reward, next_state)
        return reward
        
if __name__ == "__main__":
    try: 
        csv_data = pd.read_csv('BTCUSDT20250115.csv')
    except FileNotFoundError:
        print("error : CSV FileNotFoundError")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("eeror : CSV Empty")
        sys.exit(1)
    except pd.errors.ParserError:
        print("eeror : CSV parser error")
        sys.exit(1)
    except Exception as e:
        print(f"Exception :  {e}")
        sys.exit(1)
    max_total = 0
    epoch = 0
    print("Stored data")
    trade = TrainingEnv(csv_data)
    while True:
        print(f"epoch({epoch})")
        trade.setHyperParam(random.uniform(0,0.5),random.uniform(0.89,0.99),random.uniform(0,0.5))
        
        if epoch > 1000:
            break
        total_reward  = 0
        trade = TrainingEnv(csv_data, 100)
        for episode in range(318):
            for step in range(5000):
                reward = trade.step()
                total_reward += reward
                #print(f"Episode: {episode + 1}, Step: {step + 1}, Total Reward: {total_reward}")
        #print(f"Episode {episode + 1} finished with total reward: {total_reward}")
        if max_total > total_reward:
            max_total = total_reward
            max_alpha, max_gamma, max_epsilon = trade.getHyperParam()
            print(f"max total : {max_total},,, max alpha : {max_alpha},,, max gamma : {max_gamma},,, max epsilon : {max_epsilon}")
        total_reward = 0
        epoch+=1
