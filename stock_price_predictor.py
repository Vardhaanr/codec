import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
stocks = ['AAPL', 'MSFT', 'GOOG']
forecast_days = 30
for ticker in stocks:
    print(f"\n🔍 Processing {ticker}...")
    data = yf.download(ticker, start='2020-01-01', end='2023-01-01')
    df = data[['Close']].copy()
    df['Prediction'] = df[['Close']].shift(-forecast_days)
    X = np.array(df.drop(['Prediction'], axis=1))[:-forecast_days]
    y = np.array(df['Prediction'])[:-forecast_days]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    X_forecast = np.array(df.drop(['Prediction'], axis=1))[-forecast_days:]
    forecast = model.predict(X_forecast)
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date, periods=forecast_days + 1, freq='B')[1:]
    plt.figure(figsize=(12, 5))
    plt.plot(df['Close'], label='Historical Price', linewidth=2)
    plt.plot(future_dates, forecast, label='Predicted Price (Next 30 days)', color='red', linestyle='--')
    plt.title(f'{ticker} Stock Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
