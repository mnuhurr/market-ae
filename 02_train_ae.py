
import os
from datetime import datetime, date

import numpy as np
import pandas as pd
import tensorflow as tf

from stockstats import StockDataFrame

from common import load_settings
from models import get_autoencoder
from data import construct_sequences, construct_data



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
