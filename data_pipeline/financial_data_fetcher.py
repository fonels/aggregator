import yfinance as yf
import time
import pandas as pd
import requests

def fetch_financial_data(ticker, start_date, end_date, retries=5, delay=15):
    """
    Fetches historical financial data from Yahoo Finance for a given ticker
    with exponential backoff for retries.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'GC=F' for Gold).
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.
        retries (int): Number of times to retry in case of a failure.
        delay (int): Initial delay in seconds between retries.

    Returns:
        pandas.DataFrame: A DataFrame with the historical data, or None if it fails.
    """
    

    current_delay = delay
    for attempt in range(retries):
        try:
            # Use a session for more robust requests
            t = yf.Ticker(ticker)
            data = t.history(start=start_date, end=end_date, auto_adjust=True)
            
            if data.empty:
                print(f"Warning: No data found for {ticker} from {start_date} to {end_date}.")
                # yfinance returns an empty dataframe for valid tickers with no data in range
                return data

            if isinstance(data.columns, pd.MultiIndex):
                print(f"Warning for {ticker}: Received malformed data (MultiIndex). Retrying...")
                raise IOError("Malformed data received from yfinance API")
            
            print(f"Successfully fetched data for {ticker} from {start_date} to {end_date}.")
            # yfinance doesn't include currency, we will add it in the main pipeline
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            return data
        except Exception as e:
            print(f"Attempt {attempt + 1}/{retries} failed for {ticker}: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {current_delay} seconds...")
                time.sleep(current_delay)
                current_delay *= 2  # Double the delay for the next retry
            else:
                print(f"Could not fetch data for {ticker} after {retries} attempts.")
                return None 