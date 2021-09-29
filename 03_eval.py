
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from common import load_settings
from data import construct_sequences




def main():
    cfg = load_settings()

    cache_dir = cfg.get('cache_dir', 'data_cache')

    model_dir = cfg.get('model_dir', 'model')
    model_name = cfg.get('model_name', 'model')
    model_fn = os.path.join(model_dir, model_name)

    sequence_len = cfg.get('sequence_len', 5)

    eval_df = pd.read_csv(os.path.join(cache_dir, 'eval_data.csv'), index_col='Date')
    eval_data = np.array(eval_df)

    seqs = construct_sequences(eval_data, sequence_len)

    model = tf.keras.models.load_model(model_fn)

    seqs_pred = model(seqs)

    mse = np.mean(np.square(seqs_pred - seqs), axis=(1, 2))

    print(mse.shape, len(eval_df))

    error_df = pd.DataFrame(data={'mse': mse}, index=eval_df.index[sequence_len - 1:])
    error_df.to_csv('error.csv')

if __name__ == '__main__':
    main()