from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Flatten


def create_model(input_shape, output_size):
    input_layer = Input(shape=input_shape, name='input')
    l = Flatten(name='flatten_1')(input_layer)
    l = Dense(292, activation='relu', name='dense_1')(l)
    output_layer = Dense(output_size, activation='softmax', name='output')(l)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model
