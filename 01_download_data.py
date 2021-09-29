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

    csv_dir = os.path.join(cache_dir, 'csv')

    # make dir for the files if it's missing
    os.makedirs(csv_dir, exist_ok=True)

    # get data
    data = yfinance.download(' '.join(tickers), start=start_date, interval='1d', group_by='ticker')

    # save in separate files
    for ticker in tickers:
        df = data[ticker]
        fn = os.path.join(csv_dir, ticker + '.csv')
        df.to_csv(fn)

if __name__ == '__main__':
    main()