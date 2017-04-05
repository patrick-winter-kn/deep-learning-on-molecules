import math
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Flatten, Reshape, Dropout
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import MaxPooling1D


def create_model(input_size):
    input_layer = Input(shape=(input_size,))
    # Convolution expects 2 dimensions and will slide over the first one
    reshape_layer = Reshape((input_size, 1))(input_layer)
    # We set the window size so that every input is approximately seen by two filters
    conv_layer_1 = Convolution1D(128, math.ceil(input_size/128)*2, activation='relu')(reshape_layer)
    conv_layer_2 = Convolution1D(256, 16, activation='relu')(conv_layer_1)
    conv_layer_3 = Convolution1D(256, 8, activation='relu')(conv_layer_2)
    pool_layer = MaxPooling1D(2)(conv_layer_3)
    dropout_layer = Dropout(0.25)(pool_layer)
    flatten_layer = Flatten()(dropout_layer)
    dense_layer_1 = Dense(512, activation='relu')(flatten_layer)
    dense_layer_2 = Dense(128, activation='relu')(dense_layer_1)
    dropout_layer_2 = Dropout(0.5)(dense_layer_2)
    output_layer =Dense(2, activation='softmax')(dropout_layer_2)
    model = Model(input=input_layer, output=output_layer)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model
