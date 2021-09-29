
import os

from datetime import datetime, date
from stockstats import StockDataFrame

import numpy as np
import pandas as pd


def df_indicators(df, use_columns=None):
    """
    convert a df of a single share data into a df of to feed to the
    autoencoder

    :param df: yfinance df
    :param use_columns: list of columns to use. default is ['log_return', 'macd', 'rsi_6', 'cci', 'adx']
    :return: df of indicators/suitable data for the autoencoder
    """

    if use_columns is None:
        use_columns = ['log_return', 'macd', 'rsi_6', 'cci', 'adx']

    sdf = StockDataFrame.retype(df.copy())
    sdf['log_return'] = sdf['close'].apply(np.log).diff()
    return sdf[use_columns]


def construct_data(cfg):
    """
    construct training and evaluation datasets according to the config dict.
    returns also the index (dates) for the eval data

    :param cfg: config dict
    :return: train_data, eval_data, eval_data_index
    """
    cache_dir = cfg.get('cache_dir', 'data_cache')
    csv_dir = os.path.join(cache_dir, 'csv')

    tickers = cfg.get('tickers', [])
    split_date = cfg.get('split_date', '2020-01-01')

    # jiggle with data types. in the end we need a datetime
    if type(split_date) == str:
        split_date = datetime.fromisoformat(split_date)
    elif type(split_date) == date:
        split_date = datetime(split_date.year, split_date.month, split_date.day)

    # load all data and find common indices
    common_index = None
    df = {}

    for ticker in tickers:
        fn = os.path.join(csv_dir, ticker + '.csv')
        df[ticker] = pd.read_csv(fn, index_col='Date')
        df[ticker].index = pd.to_datetime(df[ticker].index)

        df[ticker] = df_indicators(df[ticker])
        #df[ticker].apply(np.log)

        # add some stuff
        df[ticker]['wd'] = (df[ticker].index.weekday - 2) / 2

        df[ticker].dropna(inplace=True)

        if common_index is None:
            common_index = df[ticker].index
        else:
            common_index = common_index.intersection(df[ticker].index)

    # split into train/test data
    train_data = []
    eval_data = []

    for ticker in tickers:
        current_df = df[ticker].loc[common_index]
        train_data.append(np.array(current_df.loc[common_index < split_date]))
        eval_data.append(np.array(current_df.loc[common_index >= split_date]))

    train_data = np.concatenate(train_data, axis=1)
    eval_data = np.concatenate(eval_data, axis=1)

    return train_data, eval_data, common_index[common_index >= split_date]



def construct_sequences(data, sequence_len):
    num_sequences = data.shape[0] - sequence_len + 1

    data_dim = data.shape[1:]
    seq_data_shape = (num_sequences, sequence_len) + data_dim

    seq_data = np.zeros(seq_data_shape)

    for k in range(num_sequences):
        seq_data[k] = data[k:k+sequence_len]

    return seq_data
