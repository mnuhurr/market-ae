
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from common import load_settings
from models import get_autoencoder, VariationalAutoencoder
from data import construct_sequences, construct_data


def main():
    # load and extract configuration
    cfg = load_settings()

    sequence_len = cfg.get('sequence_len', 5)
    latent_dim = cfg.get('latent_dim', 8)
    num_filters = cfg.get('num_filters', 64)
    num_conv_layers = cfg.get('num_conv_layers', 2)

    batch_size = cfg.get('batch_size', 64)
    epochs = cfg.get('epochs', 50)
    learning_rate = cfg.get('learning_rate', 1e-3)

    cache_dir = cfg.get('cache_dir', 'data_cache')
    model_dir = cfg.get('model_dir', 'model')
    model_name = cfg.get('model_name', 'model')

    model_fn = os.path.join(model_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # construct dataset from the downloaded csv files
    train_data, eval_data, eval_index = construct_data(cfg)

    # save the evaluation data for later use
    eval_df = pd.DataFrame.from_records(eval_data, index=eval_index)
    eval_df.to_csv(os.path.join(cache_dir, 'eval_data.csv'))

    # split data into sequences and create a tf dataset for training
    train_seqs = construct_sequences(train_data, sequence_len)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_seqs, train_seqs)).shuffle(4096).batch(batch_size)

    # get model
    data_dim = train_seqs.shape[1:]
    #model = get_autoencoder(data_dim, latent_dim=latent_dim, num_filters=num_filters, num_conv_layers=num_conv_layers)
    model = VariationalAutoencoder(data_dim, latent_dim=latent_dim, num_filters=num_filters, num_conv_layers=num_conv_layers)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

    model.fit(train_dataset, epochs=epochs)

    model.save(model_fn)


if __name__ == '__main__':
    main()
