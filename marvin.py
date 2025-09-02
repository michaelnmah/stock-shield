import yfinance as yf
import pandas as pd
aapl_data = yf.download('AAPl', start='2023-01-01', end='2023-12-31')
aapl_data['MA50']= aapl_data['Adj close'].rolling(window=50).mean()
aapl_data['MA200']= aapl_data['Adj close'].rolling(window=200).mean()
print(aapl_data.isnull().sum())
print(aapl_data.head())

