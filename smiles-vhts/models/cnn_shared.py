from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution1D


class SharedFeaturesModel:

    def __init__(self, input_shape, output_size, train_features_model=True):
        self.features_model, input_layer, features_layer = self.create_features_model(input_shape, train_features_model)
        self.predictions_model = self.append_predictions_model(input_layer, features_layer, output_size)
        self.predictions_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def save_features_model(self, model_file):
        self.features_model.save(model_file)

    def load_features_model(self, weights_file):
        self.features_model.load_weights(weights_file, by_name=True)

    def save_predictions_model(self, model_file):
        self.predictions_model.save(model_file)

    def load_predictions_model(self, weights_file):
        self.predictions_model.load_weights(weights_file, by_name=True)

    @staticmethod
    def create_features_model(input_shape, trainable):
        input_layer = Input(shape=input_shape, name='input')
        l = Dropout(0.2, name='dropout_input', trainable=trainable)(input_layer)
        l = Convolution1D(4, 4, activation='relu', name='convolution_1', trainable=trainable)(l)
        l = Dropout(0.2, name='dropout_convolution_1', trainable=trainable)(l)
        l = Convolution1D(16, 16, activation='relu', name='convolution_2', trainable=trainable)(l)
        l = Dropout(0.2, name='dropout_convolution_2', trainable=trainable)(l)
        l = Convolution1D(32, 32, activation='relu', name='convolution_3', trainable=trainable)(l)
        l = Dropout(0.2, name='dropout_convolution_3', trainable=trainable)(l)
        features_layer = Flatten(name='flatten_1', trainable=trainable)(l)
        model = Model(inputs=input_layer, outputs=features_layer)
        return model, input_layer, features_layer

    @staticmethod
    def append_predictions_model(input_layer, features_layer, output_size):
        l = Dense(435, activation='relu', name='dense_1')(features_layer)
        l = Dropout(0.2, name='dropout_dense_1')(l)
        l = Dense(292, activation='relu', name='dense_2')(l)
        l = Dropout(0.2, name='dropout_dense_2')(l)
        l = Dense(128, activation='relu', name='dense_3')(l)
        l = Dropout(0.2, name='dropout_dense_3')(l)
        l = Dense(64, activation='relu', name='dense_4')(l)
        l = Dropout(0.2, name='dropout_dense_4')(l)
        output_layer = Dense(output_size, activation='softmax', name='output')(l)
        model = Model(inputs=input_layer, outputs=output_layer)
        return model