import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('combined_log0.csv')
print("log File open")
csv_data = pd.read_csv('BTCUSDT20250115.csv')
print("API File open")

fig, axs = plt.subplots(2, figsize=(14,14))
axs[0].plot(csv_data.iloc[0:1590000+1]['Close'], label='Close Price')
# 거래 기록 플롯 
print("plot axs[0]")

actions = ['LONG', 'SHORT', 'EXIT LONG', 'EXIT SHORT'] 
colors = ['green', 'red', 'blue', 'purple'] 
markers = ['^', 'v', 'o', 'x'] 
for action, color, marker in zip(actions, colors, markers): 
    subset = data[data['Action'] == action] 
    axs[0].scatter(subset['Step'], subset['Price'], color=color, marker=marker, label=action)
"""
for index in range(len(data)):         
    print(index)
    if data.iloc[index]['Action'] == 'LONG': 
        axs[0].scatter(data.iloc[index]['Step'], data.iloc[index]['Price'], color='green', marker='^', label='LONG' if index == 0 else "") 
    elif data.iloc[index]['Action'] == 'SHORT':
        axs[0].scatter(data.iloc[index]['Step'], data.iloc[index]['Price'], color='red', marker='v', label='SHORT' if index == 0 else "") 
    elif data.iloc[index]['Action'] == 'EXIT LONG':
        axs[0].scatter(data.iloc[index]['Step'], data.iloc[index]['Price'], color='blue', marker='o', label='EXIT LONG' if index == 0 else "") 
    elif data.iloc[index]['Action'] == 'EXIT SHORT':
        axs[0].scatter(data.iloc[index]['Step'], data.iloc[index]['Price'], color='purple', marker='x', label='EXIT SHORT' if index == 0 else "") 
"""
print("scatter done!")

axs[0].set_xlabel('Step') 
axs[0].set_ylabel('Price') 
axs[0].set_title('Bitcoin Trading Actions') 

axs[1].plot(data['Net worth'])
axs[1].set_xlabel('Step') 
axs[1].set_ylabel('balance') 

plt.show()
