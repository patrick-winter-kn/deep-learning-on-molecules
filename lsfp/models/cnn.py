import math
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Flatten, Reshape, Dropout
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import MaxPooling1D


def create_model(input_size, output_size):
    input_layer = Input(shape=(input_size,))
    dropout_layer_1 = Dropout(0.2)(input_layer)
    # Convolution expects 2 dimensions and will slide over the first one
    reshape_layer = Reshape((input_size, 1))(dropout_layer_1)
    conv_layer_1 = Convolution1D(32, 32, activation='relu')(reshape_layer)
    dropout_layer_2 = Dropout(0.2)(conv_layer_1)
    conv_layer_2 = Convolution1D(64, 16, activation='relu')(dropout_layer_2)
    dropout_layer_3 = Dropout(0.2)(conv_layer_2)
    pool_layer = MaxPooling1D(2)(dropout_layer_3)
    dropout_layer_4 = Dropout(0.2)(pool_layer)
    flatten_layer = Flatten()(dropout_layer_4)
    dense_layer_1 = Dense(256, activation='relu')(flatten_layer)
    dropout_layer_5 = Dropout(0.2)(dense_layer_1)
    dense_layer_2 = Dense(128, activation='relu')(dropout_layer_5)
    dropout_layer_6 = Dropout(0.2)(dense_layer_2)
    output_layer =Dense(output_size, activation='softmax')(dropout_layer_6)
    model = Model(input=input_layer, output=output_layer)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


def print_model(model):
    for layer in model.layers:
        print(layer.name + ' : ' + str(layer.output_shape))
