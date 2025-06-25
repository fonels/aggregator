import requests as rq
from bs4 import BeautifulSoup
import time
import pandas as pd

class NewsFetcher:
    def __init__(self, data):
        """
        Initializes the NewsFetcher.
        Args:
            data (pd.DataFrame): A DataFrame with a 'timestamp' index and an empty 'headlines' column.
        """
        self.data = data

    def _record_headlines(self, timestamp, headlines):
        """Saves headlines for a specific timestamp."""
        # Join headlines with ' / ' if the list is not empty
        self.data.loc[timestamp, 'headlines'] = ' / '.join(headlines) if headlines else ''

    def fetch_news_for_range(self, request_delay=2):
        """
        Parses news for all timestamps in the dataframe index.

        Args:
            request_delay (int): Delay in seconds between each web request to avoid rate limiting.
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 YaBrowser/25.4.0.0 Safari/537.36'
        }

        # Timestamps are expected in the index of the dataframe
        timestamps_to_process = self.data.index.tolist()
        total_timestamps = len(timestamps_to_process)
        print(f"Starting news fetching for {total_timestamps} dates.")

        for i, timestamp in enumerate(timestamps_to_process):
            # The timestamp from financial data is a datetime object, convert to string parts.
            date_obj = pd.to_datetime(timestamp)
            year = date_obj.strftime('%Y')
            month = date_obj.strftime('%m')
            day = date_obj.strftime('%d')
            
            request_url = f'https://www.wsj.com/news/archive/{year}/{month}/{day}'
            
            try:
                request = rq.get(request_url, headers=headers, timeout=10)
                request.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

                page = BeautifulSoup(request.content, 'html.parser')

                articles = page.select('article.WSJTheme--story--XB4V2mLz, div.WSJTheme--teaser--Vxf722hl')
                
                relevant_headlines = []
                for article in articles:
                    theme_element = article.select_one('div.WSJTheme--articleType--34Gt-vdG')
                    headline_element = article.select_one('span.WSJTheme--headlineText--He1ANr9C')

                    theme_text = theme_element.get_text(strip=True) if theme_element else ""
                    headline_text = headline_element.get_text(strip=True).replace('\n', ' ') if headline_element else ""
                    
                    # List of relevant news categories
                    relevant_themes = [
                        'Major Business News', 'Economy', 'Business and Finance', 'Business and Finance - Europe', 'Business and Finance - Asia',
                        'Money & Investing', "Today's Markets", "Foreign Exchange", "Credit Markets", 'Finance', 'Business', 'Precious Metals', 'Autos Industry',
                        'China', 'U.S.', 'Russia', 'Stocks', 'Markets', 'Politics', 'Asia Economy', 'Oil Markets', 'Gas Markets', 'Tech',
                        'THE FUTURE OF EVERYTHING | WORK', 'Technology', "China's World", 'Commodities', 'Commodities Futures', 'Tech Center',
                        'Middle East', 'Law', 'Markets Main', 'Tech Stocks', 'Asia Markets', 'Europe Markets', 'U.S. Business News', 'Asian Business News',
                        'Americas Markets', 'Financing'
                    ]

                    if headline_text and theme_text in relevant_themes:
                        relevant_headlines.append(headline_text)

                self._record_headlines(timestamp, relevant_headlines)

                print(f"[{i+1}/{total_timestamps}] Successfully parsed news for {date_obj.date()}. Found {len(relevant_headlines)} headlines.")

            except rq.exceptions.RequestException as e:
                print(f"Could not fetch or parse news for {date_obj.date()}: {e}")
                self._record_headlines(timestamp, []) # Record empty on error
            
            # Delay between requests to be a good web citizen and avoid IP bans.
            time.sleep(request_delay)
        
        print("News fetching complete.")
        return self.data 