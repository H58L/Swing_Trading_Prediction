# # RNN Model
# import numpy as np
# import pandas as pd
# import yfinance as yf
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
# from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

# def get_rnn_predictions(stock_symbol="AAPL", forecast_days=5):
#     try:
#         # Fetch stock data
#         data = yf.download(stock_symbol, period="10y", interval="1d")
#         if data.empty:
#             return {"error": "No data found for this symbol"}
        
#         # Select relevant columns
#         data = data[['Open', 'High', 'Low', 'Close']]
        
#         # Separate scalers for each feature
#         scalers = {col: MinMaxScaler(feature_range=(0, 1)) for col in data.columns}
#         scaled_data = np.hstack([scalers[col].fit_transform(data[[col]]) for col in data.columns])
        
#         # Parameters
#         train_size = 0.8
#         time_steps = 30
        
#         # Split data into training and test sets
#         train_data_len = int(len(scaled_data) * train_size)
#         train_data = scaled_data[:train_data_len]
#         test_data = scaled_data[train_data_len:]
        
#         # Create datasets for OHLC prediction
#         def create_ohlc_dataset(dataset, time_steps):
#             x, y = [], []
#             for i in range(time_steps, len(dataset)):
#                 x.append(dataset[i - time_steps:i])
#                 y.append(dataset[i])  # Predict all four values (OHLC) for the next day
#             return np.array(x), np.array(y)

#         x_train, y_train = create_ohlc_dataset(train_data, time_steps)
#         x_test, y_test = create_ohlc_dataset(test_data, time_steps)
        
#         # Reshape for LSTM (samples, time steps, features)
#         x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 4))
#         x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 4))
        
#         # Manually create validation set to preserve time-series structure
#         validation_len = int(len(x_train) * 0.1)
#         x_val, y_val = x_train[-validation_len:], y_train[-validation_len:]
#         x_train, y_train = x_train[:-validation_len], y_train[:-validation_len]
        
#         # Build the LSTM model
#         model = Sequential([
#             Input(shape=(x_train.shape[1], x_train.shape[2])),
#             LSTM(units=32, return_sequences=True),
#             Dropout(0.2),
#             LSTM(units=32, return_sequences=False),
#             Dropout(0.2),
#             Dense(units=10),
#             Dense(units=4)  # Four outputs for OHLC
#         ])
        
#         # Compile the model
#         model.compile(optimizer='adam', loss='mean_squared_error')
        
#         # Early stopping and learning rate scheduler
#         early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
#         def scheduler(epoch, lr):
#             return lr * 0.9 if epoch > 10 else lr
#         lr_scheduler = LearningRateScheduler(scheduler)
        
#         # Train the model
#         model.fit(
#             x_train, y_train,
#             validation_data=(x_val, y_val),
#             batch_size=16,
#             epochs=20,
#             callbacks=[early_stopping, lr_scheduler]
#         )
        
#         # Predict OHLC prices for the test set
#         predicted_ohlc = model.predict(x_test)
        
#         # Reverse scale predictions and actual values
#         y_test_unscaled = np.hstack([
#             scalers[col].inverse_transform(y_test[:, i].reshape(-1, 1))
#             for i, col in enumerate(data.columns)
#         ])
#         predicted_ohlc_unscaled = np.hstack([
#             scalers[col].inverse_transform(predicted_ohlc[:, i].reshape(-1, 1))
#             for i, col in enumerate(data.columns)
#         ])
        
#         # Prepare response for combined chart
#         test_dates = data.index[train_data_len + time_steps:]

#         # Convert actual and predicted 'Close' values for the test set to Python lists
#         combined_dates = test_dates.strftime('%Y-%m-%d').tolist()
#         combined_actual = y_test_unscaled[:, 3].astype(float).tolist()  # Convert to float and list
#         combined_predicted = predicted_ohlc_unscaled[:, 3].astype(float).tolist()

#         # Create an array to store predictions for the next 15 days
#         next_5_days_predictions = []
#         last_sequence = scaled_data[-time_steps:]  # Use the last sequence for iterative predictions

#         for _ in range(forecast_days):
#             # Predict the next day's OHLC
#             next_day = model.predict(np.reshape(last_sequence, (1, time_steps, 4)))
            
#             # Reverse scale the predictions
#             next_day_unscaled = np.hstack([
#                 scalers[col].inverse_transform(next_day[:, i].reshape(-1, 1))
#                 for i, col in enumerate(data.columns)
#             ])
            
#             # Add to the list of predictions (convert 'Close' to float)
#             next_5_days_predictions.append(float(next_day_unscaled[0, 3]))  # Convert to float
            
#             # Update the last sequence by adding the predicted day's data (rolling window)
#             last_sequence = np.vstack([last_sequence[1:], next_day])

#         # Generate future dates for the next 15 days
#         last_date = pd.to_datetime(test_dates[-1]) if len(test_dates) > 0 else data.index[-1]
#         future_dates = (last_date + pd.to_timedelta(np.arange(1, forecast_days + 1), 'D')).strftime('%Y-%m-%d').tolist()

#         response = {
#             "dates": combined_dates,
#             "actual": combined_actual,
#             "predicted": combined_predicted,
#             "forecasted_dates": future_dates,
#             "forecasted_predictions": next_5_days_predictions
#         }

        
#         return response
#     except Exception as e:
#         return {"error": str(e)}

# import os
# import numpy as np
# import pandas as pd
# import yfinance as yf
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Bidirectional
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# import joblib

# def get_rnn_predictions(stock_symbol="RECLTD.NS", forecast_days=5):
#     try:
#         # Fetch stock data
#         data = yf.download(stock_symbol, period="max", interval="1d")
#         if data.empty:
#             return {"error": "No data found for this symbol"}
        
#         # Ensure OHLC data and include 'Volume' if available
#         if 'Volume' in data.columns:
#             data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
#         else:
#             data = data[['Open', 'High', 'Low', 'Close']]

#         # Scaler handling
#         scaler_path = "./model/rnn_scaler.pkl"
#         if os.path.exists(scaler_path):
#             scaler = joblib.load(scaler_path)
#         else:
#             scaler = MinMaxScaler(feature_range=(0, 1))
#             scaler.fit(data.values)
#             os.makedirs("model", exist_ok=True)
#             joblib.dump(scaler, scaler_path)

#         scaled_data = scaler.transform(data.values)

#         # Parameters
#         train_size = 0.8
#         time_steps = 60  # Increased time window for better context

#         # Dataset creation
#         def create_dataset(dataset, time_steps):
#             x, y = [], []
#             for i in range(time_steps, len(dataset)):
#                 x.append(dataset[i - time_steps:i])
#                 y.append(dataset[i])
#             return np.array(x), np.array(y)

#         train_data_len = int(len(scaled_data) * train_size)
#         train_data = scaled_data[:train_data_len]
#         test_data = scaled_data[train_data_len:]

#         x_train, y_train = create_dataset(train_data, time_steps)
#         x_test, y_test = create_dataset(test_data, time_steps)

#         # Reshape for LSTM
#         x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
#         x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

#         # Validation split
#         val_size = int(len(x_train) * 0.1)
#         x_val, y_val = x_train[-val_size:], y_train[-val_size:]
#         x_train, y_train = x_train[:-val_size], y_train[:-val_size]

#         model_path = "./model/rnn_model.h5"

#         # Load or train the model
#         if os.path.exists(model_path):
#             model = load_model(model_path)
#         else:
#             model = Sequential([
#                 Input(shape=(x_train.shape[1], x_train.shape[2])),
#                 Bidirectional(LSTM(50, return_sequences=True)),  # Reduced units slightly
#                 Dropout(0.2),  # Lower dropout for faster training
#                 Bidirectional(LSTM(50, return_sequences=False)),
#                 Dropout(0.2),
#                 Dense(32, activation='relu'),
#                 Dense(y_train.shape[1])
#             ])

#             model.compile(optimizer='adam', loss='mean_squared_error')

#             # Callbacks
#             early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
#             lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

#             # Train model
#             model.fit(
#                 x_train, y_train,
#                 validation_data=(x_val, y_val),
#                 batch_size=64,  # Increased batch size
#                 epochs=10,  # Reduced epochs
#                 callbacks=[early_stopping, lr_scheduler],
#                 verbose=1
#             )
#             model.save(model_path)


#         # Predictions
#         predicted_ohlc = model.predict(x_test)
#         y_test_unscaled = scaler.inverse_transform(y_test)
#         predicted_ohlc_unscaled = scaler.inverse_transform(predicted_ohlc)

#         # Future predictions
#         next_5_days_predictions = []
#         last_sequence = scaled_data[-time_steps:]

#         for _ in range(forecast_days):
#             next_day = model.predict(np.reshape(last_sequence, (1, time_steps, last_sequence.shape[1])))
#             next_day_unscaled = scaler.inverse_transform(next_day)
#             next_5_days_predictions.append(float(next_day_unscaled[0, 3]))
#             last_sequence = np.vstack([last_sequence[1:], next_day])

#         test_dates = data.index[train_data_len + time_steps:]
#         future_dates = pd.date_range(test_dates[-1], periods=forecast_days + 1)[1:].strftime('%Y-%m-%d').tolist()

#         response = {
#             "dates": test_dates.strftime('%Y-%m-%d').tolist(),
#             "actual": y_test_unscaled[:, 3].astype(float).tolist(),
#             "predicted": predicted_ohlc_unscaled[:, 3].astype(float).tolist(),
#             "forecasted_dates": future_dates,
#             "forecasted_predictions": next_5_days_predictions
#         }

#         return response

#     except Exception as e:
#         return {"error": str(e)}

import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib

def get_rnn_predictions(stock_symbol="RECLTD.NS", forecast_days=5):
    try:
        # Define paths
        model_dir = "./model"
        model_path = os.path.join(model_dir, f"{stock_symbol}_rnn_model.h5")
        scaler_path = os.path.join(model_dir, f"{stock_symbol}_rnn_scaler.pkl")
        
        # Fetch stock data
        data = yf.download(stock_symbol, period="max", interval="1d")
        if data.empty:
            return {"error": "No data found for this symbol"}
        
        # Ensure OHLC data and include 'Volume' if available
        if 'Volume' in data.columns:
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        else:
            data = data[['Open', 'High', 'Low', 'Close']]

        # Scaler handling
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(data.values)
            os.makedirs(model_dir, exist_ok=True)
            joblib.dump(scaler, scaler_path)

        scaled_data = scaler.transform(data.values)

        # Parameters
        train_size = 0.8
        time_steps = 60  # Increased time window for better context

        # Dataset creation
        def create_dataset(dataset, time_steps):
            x, y = [], []
            for i in range(time_steps, len(dataset)):
                x.append(dataset[i - time_steps:i])
                y.append(dataset[i])
            return np.array(x), np.array(y)

        train_data_len = int(len(scaled_data) * train_size)
        train_data = scaled_data[:train_data_len]
        test_data = scaled_data[train_data_len:]

        x_train, y_train = create_dataset(train_data, time_steps)
        x_test, y_test = create_dataset(test_data, time_steps)

        # Reshape for LSTM
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

        # Validation split
        val_size = int(len(x_train) * 0.1)
        x_val, y_val = x_train[-val_size:], y_train[-val_size:]
        x_train, y_train = x_train[:-val_size], y_train[:-val_size]

        # Load or train the model
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            model = Sequential([
                Input(shape=(x_train.shape[1], x_train.shape[2])),
                Bidirectional(LSTM(50, return_sequences=True)),  # Reduced units slightly
                Dropout(0.2),  # Lower dropout for faster training
                Bidirectional(LSTM(50, return_sequences=False)),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(y_train.shape[1])
            ])

            model.compile(optimizer='adam', loss='mean_squared_error')

            # Callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

            # Train model
            model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                batch_size=64,  # Increased batch size
                epochs=10,  # Reduced epochs
                callbacks=[early_stopping, lr_scheduler],
                verbose=1
            )
            model.save(model_path)

        # Predictions
        predicted_ohlc = model.predict(x_test)
        y_test_unscaled = scaler.inverse_transform(y_test)
        predicted_ohlc_unscaled = scaler.inverse_transform(predicted_ohlc)

        # Future predictions (constant for the symbol)
        next_5_days_predictions = []
        last_sequence = scaled_data[-time_steps:]

        for _ in range(forecast_days):
            next_day = model.predict(np.reshape(last_sequence, (1, time_steps, last_sequence.shape[1])))
            next_day_unscaled = scaler.inverse_transform(next_day)
            next_5_days_predictions.append(float(next_day_unscaled[0, 3]))
            last_sequence = np.vstack([last_sequence[1:], next_day])

        test_dates = data.index[train_data_len + time_steps:]
        future_dates = pd.date_range(test_dates[-1], periods=forecast_days + 1)[1:].strftime('%Y-%m-%d').tolist()

        response = {
            "dates": test_dates.strftime('%Y-%m-%d').tolist(),
            "actual": y_test_unscaled[:, 3].astype(float).tolist(),
            "predicted": predicted_ohlc_unscaled[:, 3].astype(float).tolist(),
            "forecasted_dates": future_dates,
            "forecasted_predictions": next_5_days_predictions
        }

        return response

    except Exception as e:
        return {"error": str(e)}
