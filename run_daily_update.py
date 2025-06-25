import pandas as pd
from datetime import datetime, timedelta
import psycopg2
from io import StringIO
import time
import os

# Import fetcher and uploader functions
from data_pipeline.financial_data_fetcher import fetch_financial_data
from data_pipeline.news_fetcher import NewsFetcher
from data_pipeline.config import DB_CONFIG, TABLE_MAPPINGS, METAL_TICKERS

def upload_data_to_table(connection, table_name, dataframe):
    """
    Uploads a pandas DataFrame to a specified PostgreSQL table using a temporary table
    to efficiently handle conflicts with existing data.
    """
    if dataframe.empty:
        print(f"DataFrame for {table_name} is empty. Nothing to upload.")
        return

    buffer = StringIO()
    dataframe.to_csv(buffer, index=False, header=False, sep=';')
    buffer.seek(0)
    
    columns = ','.join([f'"{col}"' for col in dataframe.columns])
    temp_table = f"temp_{table_name}_{int(time.time())}"

    try:
        with connection.cursor() as cursor:
            # Create a temporary table with the same structure as the target table
            cursor.execute(f"CREATE TEMP TABLE {temp_table} (LIKE {table_name} INCLUDING DEFAULTS);")
            
            # Copy data from the buffer to the temporary table
            copy_sql = f"COPY {temp_table} ({columns}) FROM STDIN WITH (FORMAT CSV, DELIMITER ';')"
            cursor.copy_expert(sql=copy_sql, file=buffer)
            
            # Insert data from the temporary table into the main table, ignoring duplicates
            insert_sql = f"""
                INSERT INTO {table_name} ({columns})
                SELECT {columns} FROM {temp_table}
                ON CONFLICT (timestamp) DO NOTHING;
            """
            cursor.execute(insert_sql)
            
            # Drop the temporary table
            cursor.execute(f"DROP TABLE {temp_table};")
            
            connection.commit()
            print(f"Successfully uploaded {len(dataframe)} rows to {table_name}. Ignored duplicates.")

    except Exception as e:
        print(f"An error occurred during upload to {table_name}: {e}")
        connection.rollback()

def main():
    """Main function to run the daily data collection and upload pipeline."""
    today = datetime.now() - timedelta(days=3)
    # yfinance fetches data up to the start of the given end date, so we need to go to tomorrow
    # to make sure we get all of today's data.
    tomorrow = datetime.now() + timedelta(days=1)
    
    today_str = today.strftime('%Y-%m-%d')
    tomorrow_str = tomorrow.strftime('%Y-%m-%d')

    print(f"--- Starting Daily Update for {today_str} ---")

    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("--- Successfully connected to PostgreSQL database. ---")

        # --- Step 1: Fetch and Upload Financial Data ---
        for metal, info in METAL_TICKERS.items():
            print(f"\nFetching data for {metal.upper()}...")
            # Fetch data for the day
            financial_df = fetch_financial_data(info['ticker'], today_str, tomorrow_str)

            if financial_df is not None and not financial_df.empty:
                financial_df['currency'] = info['currency']
                financial_df['unit'] = info['unit']
                
                df_to_upload = financial_df.reset_index().rename(columns={'Date': 'timestamp'})
                df_to_upload['timestamp'] = pd.to_datetime(df_to_upload['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

                table_name = TABLE_MAPPINGS[f'{metal}_data']
                upload_data_to_table(conn, table_name, df_to_upload)
            else:
                print(f"No new data for {metal.upper()} today.")
            
            # Increased delay to prevent rate limiting
            print("Waiting for 30 seconds before fetching the next metal...")
            time.sleep(30)

        # --- Step 2: Fetch and Upload News Data ---
        # News fetching is now independent of financial data fetching success
        print("\n--- Starting News Data Fetching ---")
        
        # Create a DataFrame with the target date to fetch news for.
        # This makes news fetching independent of the financial data part.
        news_index = pd.to_datetime([today_str]).tz_localize('UTC')
        news_df = pd.DataFrame(index=news_index)
        news_df['headlines'] = ''

        news_fetcher = NewsFetcher(news_df)
        news_data_with_headlines = news_fetcher.fetch_news_for_range(request_delay=3)
        
        if not news_data_with_headlines.empty and 'headlines' in news_data_with_headlines:
            # The user correctly pointed out that the fetcher might return columns 'date' and 'headlines'.
            # The logic is simplified to handle this case directly.
            news_to_upload = news_data_with_headlines.copy()
            if 'date' in news_to_upload.columns:
                news_to_upload.rename(columns={'date': 'timestamp'}, inplace=True)
            else:
                # Fallback to index if 'date' column is not present
                news_to_upload.reset_index(inplace=True)
                news_to_upload.rename(columns={'index': 'timestamp'}, inplace=True)
            
            news_to_upload['timestamp'] = pd.to_datetime(news_to_upload['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

            if 'headlines' in news_to_upload.columns and 'timestamp' in news_to_upload.columns:
                news_to_upload = news_to_upload[['timestamp', 'headlines']]
                upload_data_to_table(conn, TABLE_MAPPINGS['news_headlines'], news_to_upload)
            else:
                print("Warning: Could not prepare news data for upload (missing 'timestamp' or 'headlines').")
        else:
            print("No news found for today or failed to fetch news.")

    except psycopg2.OperationalError as e:
        print(f"Connection Error: Could not connect to the database. Please check DB_CONFIG and ensure the database is running.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("\n--- Database connection closed. ---")

if __name__ == "__main__":
    main() 