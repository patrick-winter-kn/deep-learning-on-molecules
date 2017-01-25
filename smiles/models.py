from keras import backend
from keras import objectives
from keras.models import Sequential
from keras.layers import Input, Dense, Lambda
from keras.layers.core import Dense, Flatten, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Convolution1D


def encoder(max_length, charset_length, latent_rep_size):
    model = Sequential()
    model.add(Input(shape=(max_length, charset_length), name='input_1'))
    add_encoder_layers(model, latent_rep_size)
    model.compile(optimizer='Adam', loss=calculate_loss, metrics=['accuracy'])
    return model


def decoder(max_length, charset_length, latent_rep_size):
    model = Sequential()
    model.add(Input(shape=(latent_rep_size,), name='input_1'))
    add_decoder_layers(model, max_length, charset_length, latent_rep_size)
    model.compile(optimizer='Adam', loss=calculate_loss, metrics=['accuracy'])
    return model


def autoencoder(max_length, charset_length, latent_rep_size):
    model = Sequential()
    model.add(Input(shape=(max_length, charset_length), name='input_1'))
    add_encoder_layers(model, latent_rep_size)
    add_decoder_layers(model, max_length, charset_length, latent_rep_size)
    model.compile(optimizer='Adam', loss=calculate_loss, metrics=['accuracy'])
    return model


def add_encoder_layers(model, latent_rep_size):
    model.add(Convolution1D(9, 9, activation='relu', name='conv_1'))
    model.add(Convolution1D(9, 9, activation='relu', name='conv_2'))
    model.add(Convolution1D(10, 11, activation='relu', name='conv_3'))
    model.add(Flatten(name='flatten_1'))
    model.add(Dense(435, activation='relu', name='dense_1'))
    z_mean = Dense(latent_rep_size, name='z_mean', activation='linear')
    z_log_var = Dense(latent_rep_size, name='z_log_var', activation='linear')
    model.add(Lambda(Sampler(latent_rep_size).sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var]))


def add_decoder_layers(model, max_length, charset_length, latent_rep_size):
    model.add(Dense(latent_rep_size, name='latent_input', activation='relu'))
    model.add(RepeatVector(max_length, name='repeat_vector'))
    model.add(GRU(501, return_sequences=True, name='gru_1'))
    model.add(GRU(501, return_sequences=True, name='gru_2'))
    model.add(GRU(501, return_sequences=True, name='gru_3'))
    output = Dense(charset_length, activation='softmax')
    model.add(TimeDistributed(output, name='decoded_mean'))


def calculate_loss(x, x_decoded_mean):
    # TODO max_length, z_log_var and z_mean are undefined
    x = backend.flatten(x)
    x_decoded_mean = backend.flatten(x_decoded_mean)
    x_entropy_loss = max_length * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = -0.5 * backend.mean(1 + z_log_var - backend.square(z_mean) - backend.exp(z_log_var), axis=-1)
    return x_entropy_loss + kl_loss


class Sampler:

    def __init__(self, latent_rep_size):
        self.latent_rep_size = latent_rep_size

    def sampling(self, arguments):
        epsilon_std = 0.01
        z_mean, z_log_var = arguments
        batch_size = backend.shape(z_mean)[0]
        epsilon = backend.random_normal(shape=(batch_size, self.latent_rep_size), mean=0.0, std=epsilon_std)
        return z_mean + backend.exp(z_log_var / 2) * epsilon
