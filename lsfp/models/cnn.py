import math
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Flatten, Reshape, Dropout
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import MaxPooling1D


def create_model(input_size, output_size):
    input_layer = Input(shape=(input_size,), name='input')
    l = Dropout(0.2, name='dropout_input')(input_layer)
    # Convolution expects 2 dimensions and will slide over the first one
    l = Reshape((input_size, 1), name='reshape')(l)
    l = Convolution1D(32, 32, strides=16 ,activation='relu', name='convolution_1')(l)
    l = Dropout(0.2, name='dropout_convolution_1')(l)
    l = Convolution1D(16, 16, strides=8, activation='relu', name='convolution_2')(l)
    l = Dropout(0.2, name='dropout_convolution_2')(l)
    l = MaxPooling1D(4, name='max_pooling')(l)
    l = Dropout(0.2, name='dropout_max_pooling')(l)
    l = Flatten(name='flatten')(l)
    l = Dense(128, activation='relu', name='dense_1')(l)
    l = Dropout(0.2, name='dropout_dense_1')(l)
    l = Dense(64, activation='relu', name='dense_2')(l)
    l = Dropout(0.2, name='dropout_dense_2')(l)
    output_layer =Dense(output_size, activation='softmax', name='output')(l)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


def print_model(model):
    for layer in model.layers:
        print(layer.name + ' : In:' + str(layer.input_shape[1:]) + ' Out:' + str(layer.output_shape[1:]))
