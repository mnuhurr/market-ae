
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


class VariationalAutoencoder(tf.keras.Model):
    def __init__(self, input_dim, latent_dim=8, num_filters=64, num_conv_layers=3, **kwargs):
        super(VariationalAutoencoder, self).__init__(**kwargs)

        seq_len, dim = input_dim

        # construct encoder:
        enc_input = Input(shape=input_dim)

        x = None

        for _ in range(num_conv_layers):
            if x is None:
                x = Conv1D(num_filters, kernel_size=3, activation='relu')(enc_input)
            else:
                x = Conv1D(num_filters, kernel_size=3, activation='relu')(x)
            x = BatchNormalization()(x)

        x = Flatten()(x)
        z_mu = Dense(latent_dim)(x)
        z_log_var = Dense(latent_dim)(x)

        self.encoder = tf.keras.Model(inputs=enc_input, outputs=[z_mu, z_log_var])

        self.decoder = tf.keras.models.Sequential()

        self.decoder.add(Input(shape=(latent_dim,)))
        self.decoder.add(Dense((seq_len - 2 * num_conv_layers) * num_filters))
        self.decoder.add(Reshape((seq_len - 2 * num_conv_layers, num_filters)))

        for _ in range(num_conv_layers):
            self.decoder.add(Conv1DTranspose(num_filters, kernel_size=3, activation='leaky_relu'))
            self.decoder.add(BatchNormalization())

        self.decoder.add(Conv1DTranspose(dim, kernel_size=1))

    def sample(self, z_mu, z_log_var):
        batch_size = tf.shape(z_mu)[0]
        data_dim = tf.shape(z_mu)[1]

        w = tf.random.normal(shape=(batch_size, data_dim))
        return z_mu + tf.exp(0.5 * z_log_var) * w

    def call(self, x):
        z_mu, z_log_var = self.encoder(x)
        z = self.sample(z_mu, z_log_var)
        x_hat = self.decoder(z)

        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mu) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

        self.add_loss(kl_loss)

        return x_hat


def test():
    data_dim = (10, 28)
    vae = VariationalAutoencoder(data_dim)

    vae.encoder.summary()

    vae.decoder.summary()

if __name__ == '__main__':
    test()