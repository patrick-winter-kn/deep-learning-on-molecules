from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution1D


def create_model(input_shape, output_size):
    input_layer = Input(shape=input_shape, name='input')
    l = Dropout(0.2, name='dropout_input')(input_layer)
    l = Convolution1D(9, 9, activation='relu', name='convolution_1')(l)
    l = Dropout(0.2, name='dropout_convolution_1')(l)
    l = Convolution1D(9, 9, activation='relu', name='convolution_2')(l)
    l = Dropout(0.2, name='dropout_convolution_2')(l)
    l = Convolution1D(10, 11, activation='relu', name='convolution_3')(l)
    l = Dropout(0.2, name='dropout_convolution_3')(l)
    l = Flatten(name='flatten_1')(l)
    l = Dense(435, activation='relu', name='dense_1')(l)
    l = Dropout(0.2, name='dropout_dense_1')(l)
    l = Dense(292, activation='relu', name='dense_2')(l)
    l = Dropout(0.2, name='dropout_dense_2')(l)
    l = Dense(128, activation='relu', name='dense_3')(l)
    l = Dropout(0.2, name='dropout_dense_3')(l)
    l = Dense(64, activation='relu', name='dense_4')(l)
    l = Dropout(0.2, name='dropout_dense_4')(l)
    output_layer =Dense(output_size, activation='softmax', name='output')(l)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


def print_model(model):
    for layer in model.layers:
        print(layer.name + ' : In:' + str(layer.input_shape[1:]) + ' Out:' + str(layer.output_shape[1:]))
