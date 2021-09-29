
import os
from datetime import datetime, date

import numpy as np
import pandas as pd
import tensorflow as tf

from stockstats import StockDataFrame

from common import load_settings
from models import get_autoencoder


# not sure if needed
def construct_sequences(data, sequence_len):
    num_sequences = data.shape[0] - sequence_len + 1

    data_dim = data.shape[1:]
    seq_data_shape = (num_sequences, sequence_len) + data_dim

    seq_data = np.zeros(seq_data_shape)

    for k in range(num_sequences):
        seq_data[k] = data[k:k+sequence_len]

    return seq_data


def df_indicators(df):
    sdf = StockDataFrame.retype(df.copy())
    sdf['log_return'] = sdf['close'].apply(np.log).diff()
    return sdf[['log_return', 'macd', 'rsi_6', 'cci', 'adx']]


def construct_data(cfg):
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




def main():
    cfg = load_settings()

    sequence_len = cfg.get('sequence_len', 5)
    latent_dim = cfg.get('latent_dim', 8)
    num_filters = cfg.get('num_filters', 64)

    batch_size = cfg.get('batch_size', 64)
    epochs = cfg.get('epochs', 50)
    learning_rate = cfg.get('learning_rate', 1e-3)

    cache_dir = cfg.get('cache_dir', 'data_cache')
    model_dir = cfg.get('model_dir', 'model')
    model_name = cfg.get('model_name', 'model')

    model_fn = os.path.join(model_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    train_data, eval_data, eval_index = construct_data(cfg)

    # save eval data for later use
    eval_df = pd.DataFrame.from_records(eval_data, index=eval_index)
    eval_df.to_csv(os.path.join(cache_dir, 'eval_data.csv'))

    train_seqs = construct_sequences(train_data, sequence_len)

    data_dim = train_seqs.shape[1:]

    train_dataset = tf.data.Dataset.from_tensor_slices((train_seqs, train_seqs)).shuffle(4096).batch(batch_size)

    model = get_autoencoder(data_dim, latent_dim=latent_dim, num_filters=num_filters)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

    model.summary()

    model.fit(train_dataset, epochs=epochs)

    model.save(model_fn)


if __name__ == '__main__':
    main()
