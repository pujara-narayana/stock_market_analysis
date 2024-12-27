import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Dropout

def get_stock_data(ticker, start_date, end_date):
    """
    Download stock data using yfinance
    """
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    return df

def prepare_data(df, lookback=60):
    """
    Prepare data for LSTM model
    """
    # We'll use 'Close' prices for prediction
    data = df['Close'].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences for LSTM
    X, y = [], []
    for i in range(lookback, len(data_scaled)):
        X.append(data_scaled[i-lookback:i])
        y.append(data_scaled[i])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler

def create_model(lookback):
    """
    Create LSTM model
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def plot_stock_analysis(df, ticker):
    """
    Create basic stock analysis plots
    """
    plt.figure(figsize=(15, 10))
    
    # Price trends
    plt.subplot(2, 2, 1)
    plt.plot(df.index, df['Close'])
    plt.title(f'{ticker} Stock Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    
    # Daily returns
    df['Daily Return'] = df['Close'].pct_change()
    plt.subplot(2, 2, 2)
    sns.histplot(df['Daily Return'].dropna(), kde=True)
    plt.title('Distribution of Daily Returns')
    
    # Volatility (30-day rolling standard deviation)
    df['Volatility'] = df['Daily Return'].rolling(window=30).std()
    plt.subplot(2, 2, 3)
    plt.plot(df.index, df['Volatility'])
    plt.title('30-Day Rolling Volatility')
    plt.xlabel('Date')
    
    # Volume
    plt.subplot(2, 2, 4)
    plt.plot(df.index, df['Volume'])
    plt.title('Trading Volume Over Time')
    plt.xlabel('Date')
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Parameters
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    lookback = 60  # Number of days to look back for prediction
    
    # Get data
    df = get_stock_data(ticker, start_date, end_date)
    
    # Plot analysis
    plot_stock_analysis(df, ticker)
    
    # Prepare data for LSTM
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, lookback)
    
    # Create and train model
    model = create_model(lookback)
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    # Make predictions
    predictions = model.predict(X_test)
    predictions_unscaled = scaler.inverse_transform(predictions)
    actual_unscaled = scaler.inverse_transform(y_test)
    
    # Plot predictions
    plt.figure(figsize=(12, 6))
    plt.plot(actual_unscaled, label='Actual')
    plt.plot(predictions_unscaled, label='Predicted')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.legend()
    plt.show()