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
        self.gamma = 0.97 # 할인율
        self.epsilon = 0.9 # 탐험율 exploration rate
        
        # Q-tabel init
        self.q_table = {}
        self.actions = [-2,-1,0,1,2]
        self.trades = []
        self.net_worth_Log=[]

    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        # 점진적으로 epsilon 감소
        self.epsilon = max(0.01, self.epsilon * 0.995)  # 최소 0.01까지 감소
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # 탐험
        else:
            return np.argmax(self.q_table[state])  # 최대 Q-value 선택
    def choose_action2(self, state):        
        if state not in self.q_table:
            self.q_table[state] = {a: {'q_value': 0, 'profit': 0} for a in self.actions} 
            self.epsilon = max(0.01, self.epsilon * 0.996) 
            if random.uniform(0, 1) < self.epsilon: 
                return random.choice(self.actions) 
            else: 
                return max(self.q_table[state], key=lambda x: self.q_table[state][x]['q_value'])

    def resetEnv(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.net_worth = self.initial_balance
        self.entry_price = 0
        self.trades = []
        self.net_worth_Log=[]

        
    def setHyperParam(self,alpha, gamma, epsilon):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.current_step = 0
        #print(f"alpha {self.alpha},,, gamma {self.gamma},,, epsilon {self.epsilon}")
    def getHyperParam(self):
        return self.alpha, self.gamma, self.epsilon
    def get_q_value(self, state, action):  
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))  # Q-table 초기화
        return self.q_table[state][action]
    def get_q_table(self):
        return self.q_table
        
    def update_q_value(self, state, action, reward, next_state):
        # Q-value 업데이트
        max_future_q = np.max(self.q_table[next_state]) if next_state in self.q_table else 0
        current_q = self.q_table[state][action]
        # Q-learning 업데이트 공식
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state][action] = new_q
        #self.q_table[state][action]['q_value'] = new_q
        #self.q_table[state][action]['profit'] = reward
        
    def Enter_long(self,current_price):
        if self.position == 0:
            trade_cost = self.balance*self.position_rate
            # 거래 수수료 계산 (자산*포지션 진입 비율) * 거래 수수료율
            trade_fee = trade_cost * self.fee_rate
            # 현재 자산 > 거래 수수료 : 포지션 진입 가능
            if self.balance > trade_fee + trade_cost:
                # 현재 자산 - (현재 자산*포지션 진입 비율) - 거래수수료
                self.position = trade_cost/current_price # 비트코인 long 수량
                self.balance -= (trade_cost + trade_fee)
                self.trades.append((self.current_step, current_price ,'LONG'))            
                
    def Exit_long(self, current_price):
        if self.position > 0:
            trade_value = self.position * current_price
            # 수수료 계산
            trade_cost = trade_value * self.fee_rate
            self.balance += (trade_value - trade_cost)
            self.position = 0
            self.trades.append((self.current_step, current_price,'EXIT LONG'))
            
    def Enter_short(self, current_price):
        if self.position == 0:
            trade_cost = self.balance*self.position_rate
            # 거래 수수료 계산 (자산*포지션 진입 비율) * 거래 수수료율
            trade_fee = trade_cost * self.fee_rate
            # 현재 자산 > 거래 수수료 : 포지션 진입 가능
            if self.balance > trade_fee + trade_cost:
                # 현재 자산 - (현재 자산*포지션 진입 비율) - 거래수수료
                self.position = -trade_cost/current_price # 비트코인 short 수량
                self.balance -= (trade_cost + trade_fee)
                self.entry_price = current_price
                self.trades.append((self.current_step, current_price,'SHORT'))
                
    def Exit_short(self, current_price):
        if self.position < 0:
            # 수익 = 진입가격*수량 + (진입가 - 현재가) .... 진입가 < 현재가
            entry_value = self.entry_price*abs(self.position)
            trade_value = abs(self.position) * current_price
            trade_profit = entry_value + (entry_value-trade_value)
            # 수수료 계산
            trade_cost = trade_value * self.fee_rate
            self.balance += (trade_profit - trade_cost)
            self.position = 0
            self.entry_price = 0
            self.trades.append((self.current_step, current_price,'EXIT SHORT'))
            
    def step(self):
        # now price information
        current_price = self.data.iloc[self.current_step]["Close"]
        state = (current_price, self.position) # 상태 정의
        action = self.choose_action(state)
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
            
        # 수익 = 진입가격*수량 + (진입가 - 현재가) .... 
        if self.position < 0:
            profit = (self.position*self.entry_price) + (self.position*self.entry_price - self.position*current_price)
        elif self.position > 0:
            profit = self.position*current_price
        else:
            profit = 0
            
        # 자산 업데이트
        self.net_worth = self.balance + profit
        self.net_worth_Log.append(self.net_worth)
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
    max_total = -999999999
    epoch = 0
    print("Stored data")
    trade = TrainingEnv(csv_data)
    while True:
        print(f"epoch({epoch})")
        trade.resetEnv()
        if epoch > 100:
            break
        total_reward  = 0
        #trade.setHyperParam(random.uniform(0,0.8),random.uniform(0.89,0.99),random.uniform(0,0.9))
        for episode in range(1590000): # 795*2000 , 395*2000
            reward = trade.step()
            total_reward += reward
        print(f"Episode {episode + 1} finished with total reward: {total_reward}")  
        trades_df = pd.DataFrame(trade.trades, columns=['Step', 'Price','Action'])
        net_worth_df = pd.DataFrame(trade.net_worth_Log, columns=['Net worth'])
        combined_df = trades_df.copy()
        combined_df['Net worth'] = net_worth_df
        combined_df.to_csv(f'combined_log{epoch}.csv', index=False)
        total_reward = 0
        epoch+=1
