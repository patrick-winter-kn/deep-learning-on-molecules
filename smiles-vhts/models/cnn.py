from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution1D
from keras import initializers


def create_model(input_shape, output_size):
    initializer = initializers.he_uniform()
    input_layer = Input(shape=input_shape, name='input')
    l = Dropout(0.3, name='dropout_input')(input_layer)
    l = Convolution1D(4, 4, activation='relu', name='convolution_1', kernel_initializer=initializer)(l)
    l = Dropout(0.75, name='dropout_convolution_1')(l)
    l = Convolution1D(8, 8, activation='relu', name='convolution_2', kernel_initializer=initializer)(l)
    l = Dropout(0.75, name='dropout_convolution_2')(l)
    l = Convolution1D(16, 16, activation='relu', name='convolution_3', kernel_initializer=initializer)(l)
    l = Dropout(0.75, name='dropout_convolution_3')(l)
    l = Convolution1D(32, 32, activation='relu', name='convolution_4', kernel_initializer=initializer)(l)
    l = Dropout(0.75, name='dropout_convolution_4')(l)
    l = Flatten(name='flatten_1')(l)
    l = Dense(128, activation='relu', name='dense_1', kernel_initializer=initializer)(l)
    l = Dropout(0.75, name='dropout_dense_1')(l)
    output_layer = Dense(output_size, activation='softmax', name='output', kernel_initializer=initializer)(l)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def print_model(model):
    for layer in model.layers:
        print(layer.name + ' : In:' + str(layer.input_shape[1:]) + ' Out:' + str(layer.output_shape[1:]))
