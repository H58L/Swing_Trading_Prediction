# import pandas as pd
# import numpy as np
# import yfinance as yf
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
# import tensorflow as tf

# def fetch_stock_data(stock_symbol="AAPL", lookback_period="5y"):
#     """Fetch historical stock data."""
#     stock_data = yf.download(stock_symbol, period=lookback_period, interval="1d")
#     if stock_data.empty:
#         raise ValueError("No data found for the given stock symbol.")
#     stock_data.reset_index(inplace=True)
#     return stock_data

# def preprocess_data(data, sequence_length=60):
#     """Scale and prepare data for LSTM."""
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     data_scaled = scaler.fit_transform(data.reshape(-1, 1))

#     x, y = [], []
#     for i in range(sequence_length, len(data_scaled)):
#         x.append(data_scaled[i - sequence_length:i])
#         y.append(data_scaled[i])

#     x, y = np.array(x), np.array(y)
#     return x, y, scaler

# def build_lstm_model(sequence_length):
#     """Define and compile the LSTM model."""
#     model = Sequential([
#         Input(shape=(sequence_length, 1)),
#         LSTM(50, return_sequences=True),
#         Dropout(0.2),
#         LSTM(50, return_sequences=False),
#         Dropout(0.2),
#         Dense(25),
#         Dense(1)
#     ])
#     model.compile(optimizer="adam", loss="mean_squared_error")
#     return model

# def train_lstm_model(model, x_train, y_train, x_val, y_val, epochs=10, batch_size=32):
#     """Train the LSTM model."""
#     model.fit(
#         x_train, y_train,
#         validation_data=(x_val, y_val),
#         epochs=epochs,
#         batch_size=batch_size,
#         verbose=1
#     )

# def predict_next_days(model, last_sequence, scaler, forecast_days=5):
#     """Predict the next N days based on the last known sequence."""
#     predictions = []
#     sequence = last_sequence.copy()  # Avoid modifying the original sequence

#     for _ in range(forecast_days):
#         next_day_scaled = model.predict(sequence.reshape(1, -1, 1), verbose=0)
#         next_day = scaler.inverse_transform(next_day_scaled)
#         predictions.append(next_day[0][0])
        
#         # Update the sequence with the new prediction
#         sequence = np.append(sequence[1:], next_day_scaled, axis=0)

#     return predictions

# def lstm_forecast(stock_symbol="AAPL", forecast_days=5):
#     """Main function to handle the LSTM workflow."""
#     # Step 1: Fetch and preprocess data
#     stock_data = fetch_stock_data(stock_symbol)
#     close_prices = stock_data["Close"].values
#     sequence_length = 60
#     x, y, scaler = preprocess_data(close_prices, sequence_length)

#     # Step 2: Split data into training and validation sets
#     split = int(len(x) * 0.8)
#     x_train, x_val = x[:split], x[split:]
#     y_train, y_val = y[:split], y[split:]

#     # Step 3: Build and train the model
#     model = build_lstm_model(sequence_length)
#     train_lstm_model(model, x_train, y_train, x_val, y_val)

#     # Step 4: Predict next N days
#     last_sequence = x[-1]  # Last sequence from the training data
#     future_predictions = predict_next_days(model, last_sequence, scaler, forecast_days)

#     # Step 5: Predict historical data
#     historical_predictions_scaled = model.predict(x, verbose=0)
#     historical_predictions = scaler.inverse_transform(historical_predictions_scaled)
#     historical_dates = stock_data["Date"].iloc[sequence_length:].tolist()

#     # Step 6: Prepare response with future dates
#     last_date = pd.to_datetime(stock_data["Date"].iloc[-1])
#     future_dates = [(last_date + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, forecast_days + 1)]

#     return {
#         "historical_dates": [date.strftime('%Y-%m-%d') for date in historical_dates],
#         "historical_predictions": historical_predictions.flatten().tolist(),
#         "forecasted_dates": future_dates,
#         "forecasted_predictions": future_predictions
#     }

# import os
# from flask import jsonify
# import pandas as pd
# import numpy as np
# import yfinance as yf
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
# from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
# import joblib


# # Function to fetch stock data
# def fetch_stock_data(stock_symbol="RECLTD.NS", lookback_period="10y"):
#     try:
#         stock_data = yf.download(stock_symbol, period=lookback_period)
#         if stock_data.empty:
#             raise ValueError("No data found for the given symbol and period.")
#         stock_data.reset_index(inplace=True)
#         return stock_data
#     except Exception as e:
#         print(f"Error fetching data for {stock_symbol}: {e}")
#         return None


# # Preprocess data
# def preprocess_data(close_prices, sequence_length):
#     scaler = MinMaxScaler(feature_range=(0.01, 0.99))  # Adaptive scaling
#     close_prices_scaled = scaler.fit_transform(close_prices.reshape(-1, 1))

#     x, y = [], []
#     for i in range(sequence_length, len(close_prices_scaled)):
#         x.append(close_prices_scaled[i-sequence_length:i, 0])
#         y.append(close_prices_scaled[i, 0])

#     x = np.array(x)
#     y = np.array(y)
#     x = np.reshape(x, (x.shape[0], x.shape[1], 1))  # LSTM input shape
#     return x, y, scaler


# # Build enhanced LSTM model
# def build_lstm_model(sequence_length):
#     model = Sequential([
#         Input(shape=(sequence_length, 1)),
#         LSTM(64, return_sequences=True), Dropout(0.2),
#         LSTM(64, return_sequences=True), Dropout(0.2),
#         LSTM(32, return_sequences=False), Dropout(0.2),
#         Dense(25, activation='relu'),
#         Dense(1)
#     ])
#     model.compile(optimizer="adam", loss="mean_squared_error")
#     return model


# # Train model with EarlyStopping and LearningRateScheduler
# def train_lstm_model(model, x_train, y_train, x_val, y_val, epochs=20, batch_size=32):
#     early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

#     def lr_scheduler(epoch, lr):
#         return lr * 0.9 if epoch > 10 else lr

#     model.fit(
#         x_train, y_train,
#         validation_data=(x_val, y_val),
#         epochs=epochs,
#         batch_size=batch_size,
#         callbacks=[early_stopping, LearningRateScheduler(lr_scheduler)],
#         verbose=1
#     )


# # Predict next days
# def predict_next_days(model, last_sequence, scaler, forecast_days=5):
#     predictions = []
#     sequence = last_sequence.copy()

#     for _ in range(forecast_days):
#         next_day_scaled = model.predict(sequence.reshape(1, -1, 1), verbose=0)
#         next_day = scaler.inverse_transform(next_day_scaled)
#         predictions.append(next_day[0][0])

#         sequence = np.append(sequence[1:], next_day_scaled, axis=0)
#     return predictions


# # Save model and scaler
# def save_model(model, scaler, model_path, scaler_path):
#     os.makedirs(os.path.dirname(model_path), exist_ok=True)
#     os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
#     joblib.dump(scaler, scaler_path)
#     model.save(model_path)


# # Load model and scaler
# def load_model_and_scaler(model_path, scaler_path):
#     if os.path.exists(model_path) and os.path.exists(scaler_path):
#         model = load_model(model_path)
#         scaler = joblib.load(scaler_path)
#         return model, scaler
#     return None, None


# # Main function
# def lstm_forecast(stock_symbol="RECLTD.NS", forecast_days=5, save_model_after_training=True, model_filename="./model/lstm_model.h5", scaler_filename="./model/lstm_scaler.pkl"):
#     stock_data = fetch_stock_data(stock_symbol)
#     if stock_data is None or len(stock_data) < 100:
#         return {"error": "Insufficient data to train the model."}

#     close_prices = stock_data["Close"].values
#     sequence_length = 60
#     x, y, scaler = preprocess_data(close_prices, sequence_length)

#     split = int(len(x) * 0.8)
#     x_train, x_val = x[:split], x[split:]
#     y_train, y_val = y[:split], y[split:]

#     model, loaded_scaler = load_model_and_scaler(model_filename, scaler_filename)

#     if model is None:
#         print("No pre-trained model found. Training a new model...")
#         model = build_lstm_model(sequence_length)
#         train_lstm_model(model, x_train, y_train, x_val, y_val)
#         if save_model_after_training:
#             save_model(model, scaler, model_filename, scaler_filename)
#     else:
#         scaler = loaded_scaler

#     # Predict future prices
#     last_sequence = x[-1]
#     future_predictions = predict_next_days(model, last_sequence, scaler, forecast_days)

#     # Predict historical data
#     historical_predictions_scaled = model.predict(x, verbose=0)
#     historical_predictions = scaler.inverse_transform(historical_predictions_scaled)
#     historical_dates = stock_data["Date"].iloc[sequence_length:].tolist()

#     last_date = pd.to_datetime(stock_data["Date"].iloc[-1])
#     future_dates = [(last_date + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, forecast_days + 1)]

#     return {
#         "historical_dates": [date.strftime('%Y-%m-%d') for date in historical_dates],
#         "historical_predictions": historical_predictions.flatten().tolist(),
#         "forecasted_dates": future_dates,
#         "forecasted_predictions": future_predictions
#     }

import os
from flask import jsonify
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import joblib


# Function to fetch stock data
def fetch_stock_data(stock_symbol="RECLTD.NS", lookback_period="10y"):
    try:
        stock_data = yf.download(stock_symbol, period=lookback_period)
        if stock_data.empty:
            raise ValueError("No data found for the given symbol and period.")
        stock_data.reset_index(inplace=True)
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {stock_symbol}: {e}")
        return None


# Preprocess data
def preprocess_data(close_prices, sequence_length):
    scaler = MinMaxScaler(feature_range=(0.01, 0.99))  # Adaptive scaling
    close_prices_scaled = scaler.fit_transform(close_prices.reshape(-1, 1))

    x, y = [], []
    for i in range(sequence_length, len(close_prices_scaled)):
        x.append(close_prices_scaled[i-sequence_length:i, 0])
        y.append(close_prices_scaled[i, 0])

    x = np.array(x)
    y = np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))  # LSTM input shape
    return x, y, scaler


# Build enhanced LSTM model
def build_lstm_model(sequence_length):
    model = Sequential([
        Input(shape=(sequence_length, 1)),
        LSTM(64, return_sequences=True), Dropout(0.2),
        LSTM(64, return_sequences=True), Dropout(0.2),
        LSTM(32, return_sequences=False), Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# Train model with EarlyStopping and LearningRateScheduler
def train_lstm_model(model, x_train, y_train, x_val, y_val, epochs=20, batch_size=32):
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    def lr_scheduler(epoch, lr):
        return lr * 0.9 if epoch > 10 else lr

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, LearningRateScheduler(lr_scheduler)],
        verbose=1
    )


# Predict next days
def predict_next_days(model, last_sequence, scaler, forecast_days=5):
    predictions = []
    sequence = last_sequence.copy()

    for _ in range(forecast_days):
        next_day_scaled = model.predict(sequence.reshape(1, -1, 1), verbose=0)
        next_day = scaler.inverse_transform(next_day_scaled)
        predictions.append(next_day[0][0])

        sequence = np.append(sequence[1:], next_day_scaled, axis=0)
    return predictions


# Save model and scaler for specific symbol
def save_model(model, scaler, stock_symbol, model_dir="./model"):
    os.makedirs(model_dir, exist_ok=True)
    model_filename = os.path.join(model_dir, f"{stock_symbol}_lstm_model.h5")
    scaler_filename = os.path.join(model_dir, f"{stock_symbol}_lstm_scaler.pkl")
    
    # Save model and scaler with the symbol-specific filenames
    joblib.dump(scaler, scaler_filename)
    model.save(model_filename)


# Load model and scaler for specific symbol
def load_model_and_scaler(stock_symbol, model_dir="./model"):
    model_filename = os.path.join(model_dir, f"{stock_symbol}_lstm_model.h5")
    scaler_filename = os.path.join(model_dir, f"{stock_symbol}_lstm_scaler.pkl")
    
    if os.path.exists(model_filename) and os.path.exists(scaler_filename):
        model = load_model(model_filename)
        scaler = joblib.load(scaler_filename)
        return model, scaler
    return None, None


# Main function
def lstm_forecast(stock_symbol="RECLTD.NS", forecast_days=5, save_model_after_training=True, model_dir="./model"):
    stock_data = fetch_stock_data(stock_symbol)
    if stock_data is None or len(stock_data) < 100:
        return {"error": "Insufficient data to train the model."}

    close_prices = stock_data["Close"].values
    sequence_length = 60
    x, y, scaler = preprocess_data(close_prices, sequence_length)

    split = int(len(x) * 0.8)
    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]

    model, loaded_scaler = load_model_and_scaler(stock_symbol, model_dir)

    if model is None:
        print("No pre-trained model found. Training a new model...")
        model = build_lstm_model(sequence_length)
        train_lstm_model(model, x_train, y_train, x_val, y_val)
        if save_model_after_training:
            save_model(model, scaler, stock_symbol, model_dir)
    else:
        scaler = loaded_scaler

    # Predict future prices
    last_sequence = x[-1]
    future_predictions = predict_next_days(model, last_sequence, scaler, forecast_days)

    # Predict historical data
    historical_predictions_scaled = model.predict(x, verbose=0)
    historical_predictions = scaler.inverse_transform(historical_predictions_scaled)
    historical_dates = stock_data["Date"].iloc[sequence_length:].tolist()

    last_date = pd.to_datetime(stock_data["Date"].iloc[-1])
    future_dates = [(last_date + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, forecast_days + 1)]

    return {
        "historical_dates": [date.strftime('%Y-%m-%d') for date in historical_dates],
        "historical_predictions": historical_predictions.flatten().tolist(),
        "forecasted_dates": future_dates,
        "forecasted_predictions": future_predictions
    }
