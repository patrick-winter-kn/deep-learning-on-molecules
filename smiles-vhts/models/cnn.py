from keras import backend
from keras.models import Model
from keras.layers import Input, Lambda
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Convolution1D


def create_model(input_shape, output_size, latent_rep_size):
    input_layer = Input(shape=input_shape, name='input')
    l = Convolution1D(9, 9, activation='relu', name='convolution_1')(input_layer)
    l = Convolution1D(9, 9, activation='relu', name='convolution_2')(l)
    l = Convolution1D(10, 11, activation='relu', name='convolution_3')(l)
    l = Flatten(name='flatten_1')(l)
    l = Dense(435, activation='relu', name='dense_1')(l)
    z_mean = Dense(latent_rep_size, name='z_mean', activation='linear')(l)
    z_log_var = Dense(latent_rep_size, name='z_log_var', activation='linear')(l)
    l = Lambda(Sampler(latent_rep_size).sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var])
    l = Dense(128, activation='relu', name='dense_2')(l)
    l = Dense(64, activation='relu', name='dense_3')(l)
    output_layer =Dense(output_size, activation='softmax', name='output')(l)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


def print_model(model):
    for layer in model.layers:
        print(layer.name + ' : In:' + str(layer.input_shape[1:]) + ' Out:' + str(layer.output_shape[1:]))


class Sampler:

    def __init__(self, latent_rep_size):
        self.latent_rep_size = latent_rep_size

    def sampling(self, arguments):
        epsilon_std = 0.01
        z_mean, z_log_var = arguments
        batch_size = backend.shape(z_mean)[0]
        epsilon = backend.random_normal(shape=(batch_size, self.latent_rep_size), mean=0.0, stddev=epsilon_std)
        return z_mean + backend.exp(z_log_var / 2) * epsilon
