# market-ae
train and evaluate a simple autoencoder to detect market anomalies. ongoing work. 

currently the implemented autoencoders are based on 1d convolution.


TODO:
- fancier model(s)
- reassess data columns
- draw some plots
- ...


### settings.yaml

directory config:
- cache_dir
- models_dir
- results_dir

dates:
- start_date: startpoint of the data to download
- split_date: everything before split_date is training data, everything after is evaluation data

example settings.yaml:
```
cache_dir: data_cache
models_dir: models
results_dir: results

tickers:
  - AAPL
  - AMZN
  - MSFT
  - GOOG

start_date: 2000-01-01
split_date: 2019-01-01

model_name: autoencoder

sequence_len: 10

latent_dim: 16
num_filters: 256
num_conv_layers: 3

batch_size: 512
epochs: 200
learning_rate: 0.001
```
