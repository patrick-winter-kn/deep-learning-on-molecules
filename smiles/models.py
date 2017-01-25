from keras import backend
from keras import objectives
from keras.models import Model
from keras.layers import Input, Lambda
from keras.layers.core import Dense, Flatten, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Convolution1D


def encoder(max_length, charset_length, latent_rep_size):
    inputs = Input(shape=(max_length, charset_length), name='input_1')
    outputs, loss_function = add_encoder_layers(inputs, max_length, latent_rep_size)
    model = Model(input=inputs, output=outputs)
    model.compile(optimizer='Adam', loss=loss_function, metrics=['accuracy'])
    return model


def decoder(max_length, charset_length, latent_rep_size):
    inputs = Input(shape=(latent_rep_size,), name='input_1')
    outputs = add_decoder_layers(inputs, max_length, charset_length, latent_rep_size)
    model = Model(input=inputs, output=outputs)
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def autoencoder(max_length, charset_length, latent_rep_size):
    inputs = Input(shape=(max_length, charset_length), name='input_1')
    latent_outputs, loss_function = add_encoder_layers(inputs, max_length, latent_rep_size)
    outputs = add_decoder_layers(latent_outputs, max_length, charset_length, latent_rep_size)
    model = Model(input=inputs, output=outputs)
    model.compile(optimizer='Adam', loss=loss_function, metrics=['accuracy'])
    return model


def add_encoder_layers(inputs, max_length, latent_rep_size):
    l = Convolution1D(9, 9, activation='relu', name='conv_1')(inputs)
    l = Convolution1D(9, 9, activation='relu', name='conv_2')(l)
    l = Convolution1D(10, 11, activation='relu', name='conv_3')(l)
    l = Flatten(name='flatten_1')(l)
    l = Dense(435, activation='relu', name='dense_1')(l)
    z_mean = Dense(latent_rep_size, name='z_mean', activation='linear')(l)
    z_log_var = Dense(latent_rep_size, name='z_log_var', activation='linear')(l)
    l = Lambda(Sampler(latent_rep_size).sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var])
    return l, LossCalculator(max_length, z_log_var, z_mean).calculate_loss


def add_decoder_layers(inputs, max_length, charset_length, latent_rep_size):
    l = Dense(latent_rep_size, name='latent_input', activation='relu')(inputs)
    l = RepeatVector(max_length, name='repeat_vector')(l)
    l = GRU(501, return_sequences=True, name='gru_1')(l)
    l = GRU(501, return_sequences=True, name='gru_2')(l)
    l = GRU(501, return_sequences=True, name='gru_3')(l)
    l = TimeDistributed(Dense(charset_length, activation='softmax'), name='decoded_mean')(l)
    return l


class LossCalculator:

    def __init__(self, max_length, z_log_var, z_mean):
        self.max_length = max_length
        self.z_log_var = z_log_var
        self.z_mean = z_mean

    def calculate_loss(self, x, x_decoded_mean):
        x = backend.flatten(x)
        x_decoded_mean = backend.flatten(x_decoded_mean)
        x_entropy_loss = self.max_length * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = -0.5 * backend.mean(1 + self.z_log_var - backend.square(self.z_mean) - backend.exp(self.z_log_var),
                                      axis=-1)
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
