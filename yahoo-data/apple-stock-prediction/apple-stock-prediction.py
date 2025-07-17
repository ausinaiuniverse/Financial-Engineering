import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- Parameters ---
ticker = 'AAPL'
start_date = '2015-01-01'
end_date = '2024-12-31'
window_size = 60
epochs = 20
batch_size = 32

# --- 1. Download Data ---
df = yf.download(ticker, start=start_date, end=end_date)
data = df[['Close']]

# --- 2. Preprocess ---
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

def create_sequences(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, window_size)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# --- 3. Build LSTM Model ---
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# --- 4. Train Model ---
model.fit(X, y, epochs=epochs, batch_size=batch_size)

# --- 5. Predict and Inverse Transform ---
predictions = model.predict(X)
predictions = scaler.inverse_transform(predictions)
actual_prices = scaler.inverse_transform(y.reshape(-1, 1))

# --- 6. Plot ---
plt.figure(figsize=(14, 6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predictions, label='Predicted Price')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
