
import tensorflow as tf

from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Flatten, Dense, Conv1DTranspose, Reshape


def get_autoencoder(input_dim, latent_dim=8, num_filters=64, num_conv_layers=3):
    """
    create a simple 1d convolutional autoencoder. kernel size is fixed to 3

    :param input_dim: data dimension, should be in the form (sequence_len, num_features)
    :param latent_dim: size of the latent feature layer
    :param num_filters: number of convolutional filters
    :param num_conv_layers: number of convolutional layers in the encoder/decoder
    :return: keras model
    """

    seq_len, dim = input_dim

    model = tf.keras.models.Sequential()

    model.add(Input(shape=input_dim))

    for _ in range(num_conv_layers):
        model.add(Conv1D(num_filters, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(latent_dim))

    model.add(Dense((seq_len - 2 * num_conv_layers) * num_filters))
    model.add(Reshape((seq_len - 2 * num_conv_layers, num_filters)))

    for _ in range(num_conv_layers):
        model.add(Conv1DTranspose(num_filters, kernel_size=3, activation='leaky_relu'))
        model.add(BatchNormalization())

    model.add(Conv1DTranspose(dim, kernel_size=1))

    return model


if __name__ == '__main__':
    test()