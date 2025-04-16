import yfinance as yf
import pandas as pd
import numpy as np
from news_parse import NewsParse

class PriceHistory:
    def __init__(self, data, ticker = 'GC=F', period = '5d', currency = 'usd', unit = 'ounce'):
        self.gold = yf.Ticker(ticker)
        self.hist = pd.DataFrame(self.gold.history(period=period))
        self.currency = currency
        self.unit = unit
        self.data = data

    def get_price_hist(self):
        price_format = lambda x: np.round(x,3)

        self.hist['timestamp'] = self.hist.index.astype(str).str[:10]
        self.hist = self.hist.set_index('timestamp')
        self.data = self.data.set_index('timestamp')
        self.data['open'] = self.hist['Open'].astype(float).apply(price_format)
        self.data['high'] = self.hist['High'].astype(float).apply(price_format)
        self.data['low'] = self.hist['Low'].astype(float).apply(price_format)
        self.data['close'] = self.hist['Close'].astype(float).apply(price_format)
        self.data['volume'] = self.hist['Volume']
        self.data['currency'] = self.currency
        self.data['unit'] =  self.unit

        if 'headlines' not in self.data.columns:
             self.data['headlines'] = ''

        news_parse_instance = NewsParse(self.data)
        news_parse_instance.get_parsed_news()
        self.data = news_parse_instance.data
        self.data.reset_index().to_csv('platinum_final_data.csv', index=False, sep=';')


if __name__ == "__main__":
    initial_data = pd.read_csv('gold_final_data.csv', sep=';')
    price_hist_processor = PriceHistory(initial_data, period='50y', ticker='PL=F')
    price_hist_processor.get_price_hist()