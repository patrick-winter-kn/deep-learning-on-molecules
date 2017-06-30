from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Flatten, Dropout
from keras import initializers


def create_model(input_shape, output_size):
    initializer = initializers.he_uniform()
    input_layer = Input(shape=input_shape, name='input')
    l = Dropout(0.3, name='dropout_input')(input_layer)
    l = Flatten(name='flatten_1')(l)
    l = Dense(836, activation='relu', name='dense_1', kernel_initializer=initializer)(l)
    l = Dropout(0.75, name='dropout_dense_1')(l)
    l = Dense(1616, activation='relu', name='dense_2', kernel_initializer=initializer)(l)
    l = Dropout(0.75, name='dropout_dense_2')(l)
    l = Dense(2992, activation='relu', name='dense_3', kernel_initializer=initializer)(l)
    l = Dropout(0.75, name='dropout_dense_3')(l)
    l = Dense(4992, activation='relu', name='dense_4', kernel_initializer=initializer)(l)
    l = Dropout(0.75, name='dropout_dense_4')(l)
    l = Dense(128, activation='relu', name='dense_5', kernel_initializer=initializer)(l)
    l = Dropout(0.75, name='dropout_dense_5')(l)
    output_layer = Dense(output_size, activation='softmax', name='output', kernel_initializer=initializer)(l)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model
