import requests as rq
from bs4 import BeautifulSoup
import time
import pandas as pd

class NewsParse:
    def __init__(self, data):
        self.data = data

    def record_data(self, timestamp, headlines):
        self.data.loc[timestamp, 'headlines'] = headlines

    def get_parsed_news(self, start_timestamp=None):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        all_timestamps = self.data.index.tolist()
        timestamps_to_process = [timestamp for timestamp in all_timestamps if pd.isna(self.data.loc[timestamp, 'headlines'])]

        if start_timestamp:
            try:
                start_index = all_timestamps.index(start_timestamp)
                timestamps_to_process = all_timestamps[start_index:]
                print(f'Continuing parsing from timestamp: {start_timestamp}')
            except ValueError:
                print(f'Warning: start_timestamp {start_timestamp} not found in data index. Processing all timestamps.')

        counter = counter_last = 0

        for timestamp in timestamps_to_process:
            year = timestamp[:4]
            month = timestamp[5:7]
            day = timestamp[8:10]
            request_url = f'https://www.wsj.com/news/archive/{year}/{month}/{day}'
            request = rq.get(request_url, headers=headers)

            page = BeautifulSoup(request.content,'html.parser')

            articles = page.select('article.WSJTheme--story--XB4V2mLz')
            if not articles:
                 articles = page.select('div.WSJTheme--teaser--Vxf722hl')

            relevant_headlines = []
            for article in articles:
                 theme_element = article.select_one('div.WSJTheme--articleType--34Gt-vdG')
                 headline_element = article.select_one('span.WSJTheme--headlineText--He1ANr9C')

                 theme_text = theme_element.get_text(strip=True) if theme_element else ""
                 headline_text = headline_element.get_text(strip=True).replace('\n', ' ') if headline_element else ""

                 if headline_text and theme_text in [
                     'Major Business News', 'Economy', 'Business and Finance', 'Business and Finance - Europe', 'Business and Finance - Asia',
                     'Money & Investing', "Today's Markets", "Foreign Exchange", "Credit Markets", 'Finance', 'Business', 'Precious Metals', 'Autos Industry',
                     'China', 'U.S.', 'Russia', 'Stocks', 'Markets', 'Politics', 'Asia Economy', 'Oil Markets', 'Gas Markets', 'Tech',
                     'THE FUTURE OF EVERYTHING | WORK', 'Technology', 'China’s World', 'Commodities', 'Commodities Futures', 'Tech Center',
                 'Middle East', 'Law', 'Markets Main', 'Tech Stocks', 'Asia Markets', 'Europe Markets', 'U.S. Business News', 'Asian Business News',
                 'Americas Markets', 'Financing']:
                     relevant_headlines.append(headline_text)

            if relevant_headlines:
                self.record_data(timestamp, ' / '.join(relevant_headlines))
            else:
                self.record_data(timestamp, '')

            counter += 1
            if counter - counter_last == 100:
                print(f'Dataset saved on {timestamp}!')
                counter_last = counter
                self.data.reset_index().to_csv('intermediate_results.csv', index=False, sep=';')

            time.sleep(1)