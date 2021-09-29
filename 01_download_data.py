"""
download data according to the settings file

"""

import os
import pandas as pd
import yfinance

from common import load_settings



def main():
    cfg = load_settings()

    cache_dir = cfg.get('cache_dir', 'data_cache')
    tickers = cfg.get('tickers', [])
    start_date = cfg.get('start_date', '2000-01-01')

    os.makedirs(cache_dir, exist_ok=True)

    data = yfinance.download(' '.join(tickers), start=start_date, interval='1d', group_by='ticker')

    for ticker in tickers:
        df = data[ticker]
        fn = os.path.join(cache_dir, ticker + '.csv')
        df.to_csv(fn)

if __name__ == '__main__':
    main()