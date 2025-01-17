import pandas as pd

data = pd.read_csv('BTCUSDT20250115.csv')
print(data.iloc[1]['Close'])