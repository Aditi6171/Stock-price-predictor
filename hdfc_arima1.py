import numpy as np
import datetime
import yfinance as yf
import warnings
from scipy import stats
from statsmodels.stats.descriptivestats import describe
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader as pdr
from pandas_datareader import data, wb
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error

# Ignore warnings
warnings.filterwarnings('ignore')

# Download Data from Yahoo Finance
start = datetime.datetime(2012, 12, 31)
end = datetime.datetime(2022, 12, 31)
df = yf.download("HDFCBANK.NS", start=start, end=end, interval='1mo')

df['PCT Returns'] = df['Adj Close'].pct_change()
df['Log Returns'] = np.log(df['PCT Returns'] + 1)
df['Log Cumsum'] = df['Log Returns'].cumsum()
df = df[['Adj Close', 'PCT Returns', 'Log Returns', 'Log Cumsum']].copy()
df.dropna(inplace=True)
print(df)
# cols = df.columns
# for col in cols:
#     sns.lineplot(df[col])
# plt.show()
# plt.plot(df['Adj Close'])
# plt.show()
# plt.plot(df['PCT Returns'])
# plt.show()

# split into train and test sets
rmse = []
df1 = df['Adj Close'].dropna()
test_size = 36
size = len(df1) - test_size
train, test = df1[0:size], df1[size:len(df1)]
model = auto_arima(train, start_p=0, start_q=0)
print(model.summary())
output = model.predict(n_periods=len(test))
test = pd.DataFrame(test)
test['yhat'] = output
test['yhat'].fillna(0)
print(test)
obs = test
rmse = np.sqrt(mean_squared_error(test['Adj Close'], test['yhat']))
print('Test RMSE: %.3f' % rmse)
cols = test.columns
for col in cols:
    sns.lineplot(test[col])
plt.show()
