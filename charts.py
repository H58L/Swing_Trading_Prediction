import yfinance as yf
import plotly.graph_objects as go
import json
import pandas as pd

def create_candlestick_chart(stock_symbol="RECLTD.NS", period="5y"):
    # Fetch data for the past 5 years
    stock_data = yf.download(stock_symbol, period=period, interval="1d")
    
    # Debugging: Print the fetched data
    print(stock_data)
    
    # Ensure we have data
    if stock_data.empty:
        return {"error": f"No data found for the symbol '{stock_symbol}'."}
    
    # Reset index to make 'Date' a column
    stock_data.reset_index(inplace=True)

    # Debugging: Check column names and first few rows
    print(stock_data.columns)
    print(stock_data.head())

    # Extract the necessary columns dynamically, checking if symbol is in the column names
    date_col = 'Date'
    open_col = [col for col in stock_data.columns if 'Open' in col][0]  # Find column containing 'Open'
    high_col = [col for col in stock_data.columns if 'High' in col][0]  # Find column containing 'High'
    low_col = [col for col in stock_data.columns if 'Low' in col][0]  # Find column containing 'Low'
    close_col = [col for col in stock_data.columns if 'Close' in col][0]  # Find column containing 'Close'
    
    # Extract individual columns based on dynamic names
    dates = stock_data[date_col].astype(str)  # Convert dates to strings for JSON compatibility
    open_prices = stock_data[open_col]
    high_prices = stock_data[high_col]
    low_prices = stock_data[low_col]
    close_prices = stock_data[close_col]

    # Create a candlestick chart
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=dates,
                open=open_prices,
                high=high_prices,
                low=low_prices,
                close=close_prices
            )
        ]
    )

    # Add title and layout configurations
    fig.update_layout(
        title=f"{stock_symbol.upper()} Candlestick Chart ({period})",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=True,
        template="plotly_dark"
    )

    # Convert the figure to JSON for API response
    return json.loads(fig.to_json())