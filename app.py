from flask import Flask, jsonify, request
from sklearn.preprocessing import MinMaxScaler
from charts import create_candlestick_chart
from RNN import get_rnn_predictions
from LSTM import build_lstm_model, fetch_stock_data, load_model_and_scaler, preprocess_data, predict_next_days, lstm_forecast, save_model, train_lstm_model
import joblib
import os
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)

@app.route('/candlestick-chart', methods=['GET'])
def candlestick_chart():
    # Get stock symbol from query parameter (default: RECLTD.NS)
    stock_symbol = request.args.get('symbol', 'RECLTD.NS')
    period = request.args.get('period', '5y')  # Default period
    candlestick_data = create_candlestick_chart(stock_symbol, period)
    return jsonify(candlestick_data)

@app.route('/rnn', methods=['GET'])
def combined_chart_rnn():
    stock_symbol = request.args.get('symbol', 'RECLTD.NS')
    period = request.args.get('period', '5y')  # Default period
    forecast_days = int(request.args.get('forecast_days', 5))  # Default forecast days
    
    # Generate candlestick data
    candlestick_data = create_candlestick_chart(stock_symbol, period)
    if "error" in candlestick_data:
        return jsonify(candlestick_data)
    
    # Get RNN predictions
    rnn_data = get_rnn_predictions(stock_symbol, forecast_days=forecast_days)
    if "error" in rnn_data:
        return jsonify(rnn_data)
    
    # Filter predictions to match the selected period
    candlestick_dates = [
        item['x'] 
        for item in candlestick_data["data"] 
        if item["type"] == "candlestick"
    ][0]  # Extract dates from the candlestick chart
    start_date = pd.to_datetime(candlestick_dates[0]).tz_localize(None)
    end_date = pd.to_datetime(candlestick_dates[-1]).tz_localize(None)

    # Filter RNN dates and predictions within the candlestick period
    filtered_predictions = [
        (date, value) 
        for date, value in zip(rnn_data["dates"], rnn_data["predicted"])
        if start_date <= pd.to_datetime(date).tz_localize(None) <= end_date
    ]
    filtered_dates, filtered_values = zip(*filtered_predictions) if filtered_predictions else ([], [])
    
    # Add RNN prediction data to the chart
    combined_data = candlestick_data
    combined_data["data"].append({
        "x": filtered_dates,
        "y": filtered_values,
        "type": "scatter",
        "mode": "lines",
        "name": "RNN Predictions",
        "line": {"color": "blue", "width": 2}
    })

    # Add forecasted predictions (next 5 days)
    combined_data["data"].append({
        "x": rnn_data["forecasted_dates"],
        "y": rnn_data["forecasted_predictions"],
        "type": "scatter",
        "mode": "lines",
        "name": "Forecasted Predictions",
        "line": {"color": "green", "width": 2, "dash": "dash"}
    })

    # Add table data for forecasted predictions
    combined_data["table_data"] = {
        "forecasted_dates": rnn_data["forecasted_dates"],
        "forecasted_predictions": rnn_data["forecasted_predictions"]
    }

    return jsonify(combined_data)

# Define paths for the saved model and scaler
MODEL_FILENAME = "./model/lstm_model.h5"  # Use .h5 to save the model as a Keras model
SCALER_FILENAME = "./model/lstm_scaler.pkl"

def get_model_paths(stock_symbol):
    """Use global model and scaler."""
    model_path = "./model/lstm_modeL.h5"  # Global model for all symbols
    scaler_path = "./model/lstm_scaler.pkl"    # Global scaler for all symbols
    return model_path, scaler_path

@app.route('/lstm', methods=['GET'])
def combined_chart_lstm():
    stock_symbol = request.args.get('symbol', 'RECLTD.NS')
    period = request.args.get('period', '5y')  # Default period for candlestick data
    forecast_days = int(request.args.get('forecast_days', 5))

    try:
        # Fetch stock data with 10-year lookback period as per updated lstm.py
        stock_data = fetch_stock_data(stock_symbol, lookback_period='10y')
        if stock_data is None or stock_data.empty:
            return jsonify({"error": f"No data found for stock symbol {stock_symbol} in the given period."})

        close_prices = stock_data["Close"].values
        sequence_length = 60

        # Define model and scaler file paths based on stock symbol
        model_dir = "./model"
        model_filename = os.path.join(model_dir, f"{stock_symbol}_lstm_model.h5")
        scaler_filename = os.path.join(model_dir, f"{stock_symbol}_lstm_scaler.pkl")

        # Load or create model and scaler for the symbol
        model, scaler = load_model_and_scaler(stock_symbol, model_dir)

        if model is None or scaler is None:
            print(f"Model and scaler for {stock_symbol} not found. Training a new model.")
            # Train a new model if not found
            model = build_lstm_model(sequence_length)
            x, y, scaler = preprocess_data(close_prices, sequence_length)
            train_lstm_model(model, x, y, x, y)  # Train with the data
            save_model(model, scaler, stock_symbol, model_dir)  # Save the new model
            print(f"New model and scaler for {stock_symbol} have been trained and saved.")
        else:
            print(f"Loaded pre-trained model for {stock_symbol}.")

        # Use the last sequence for predictions
        x, y, scaler = preprocess_data(close_prices, sequence_length)  # Re-preprocess data for prediction
        last_sequence = x[-1]
        future_predictions = predict_next_days(model, last_sequence, scaler, forecast_days)

        # Prepare historical predictions
        historical_predictions_scaled = model.predict(x, verbose=0)
        historical_predictions = scaler.inverse_transform(historical_predictions_scaled)
        historical_dates = stock_data["Date"].iloc[sequence_length:].values  # Start from sequence_length index

        # Ensure lengths match for dates and predictions
        if len(historical_dates) != len(historical_predictions):
            raise ValueError("Mismatch in lengths of historical dates and predictions.")

        # Calculate future dates based on last known date
        last_date = pd.to_datetime(stock_data["Date"].iloc[-1])
        future_dates = [(last_date + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, forecast_days + 1)]

        # Combine candlestick data with LSTM forecast (assuming create_candlestick_chart is defined elsewhere)
        candlestick_data = create_candlestick_chart(stock_symbol, period)
        if "error" in candlestick_data:
            return jsonify(candlestick_data)

        # Filter historical predictions based on the candlestick period
        candlestick_dates = [
            pd.to_datetime(item['x']).tz_localize(None) 
            for item in candlestick_data['data'] if item['type'] == 'candlestick'
        ][0]
        start_date = candlestick_dates[0]
        end_date = candlestick_dates[-1]

        # Filter historical predictions based on the selected period
        historical_predictions_filtered = [
            (str(date), float(value)) 
            for date, value in zip(historical_dates, historical_predictions.flatten())
            if start_date <= pd.to_datetime(date) <= end_date
        ]

        # Prepare filtered historical dates and values
        historical_dates_filtered, historical_values_filtered = zip(*historical_predictions_filtered)

        # Forecasted Predictions
        forecasted_dates = [str(date) for date in future_dates]
        forecasted_values = [float(value) for value in future_predictions]

        # Combine candlestick data with LSTM forecast data
        combined_data = candlestick_data
        combined_data["data"].append({
            "x": historical_dates_filtered,
            "y": historical_values_filtered,
            "type": "scatter",
            "mode": "lines",
            "name": "LSTM Predictions (Historical)",
            "line": {"color": "blue", "width": 2}
        })

        # Add forecasted data
        combined_data["data"].append({
            "x": forecasted_dates,
            "y": forecasted_values,
            "type": "scatter",
            "mode": "lines",
            "name": "LSTM Predictions (Forecast)",
            "line": {"color": "green", "width": 2, "dash": "dash"}
        })

        # Include forecast table data
        combined_data["table_data"] = {
            "forecasted_dates": forecasted_dates,
            "forecasted_predictions": forecasted_values
        }

        return jsonify(combined_data)

    except Exception as e:
        print(f"Error during LSTM forecast: {e}")
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)