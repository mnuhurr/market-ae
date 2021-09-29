
import tensorflow as tf

from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Flatten, Dense, Conv1DTranspose, Reshape


def get_autoencoder(input_dim, latent_dim=8, num_filters=64):

    seq_len, dim = input_dim

    model = tf.keras.models.Sequential()

    model.add(Input(shape=input_dim))

    model.add(Conv1D(num_filters, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(num_filters, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(num_filters, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(latent_dim))

    model.add(Dense((seq_len - 6) * num_filters))
    model.add(Reshape((seq_len - 6, num_filters)))

    model.add(Conv1DTranspose(num_filters, kernel_size=3, activation='leaky_relu'))
    model.add(BatchNormalization())

    model.add(Conv1DTranspose(num_filters, kernel_size=3, activation='leaky_relu'))
    model.add(BatchNormalization())

    model.add(Conv1DTranspose(num_filters, kernel_size=3, activation='leaky_relu'))
    model.add(BatchNormalization())

    model.add(Conv1DTranspose(dim, kernel_size=1))

    return model

def test():
    data_dim = (10, 28)

    model = get_autoencoder(data_dim)
    model.summary()

    print(model.input_shape)
    print(model.output_shape)

if __name__ == '__main__':
    test()