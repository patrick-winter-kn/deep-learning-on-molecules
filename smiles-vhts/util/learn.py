from models import cnn
import h5py
from os import path
from keras import models
from keras.callbacks import  ModelCheckpoint, ReduceLROnPlateau, Callback


def train(train_file, model_file, epochs, batch_size):
    train_hdf5 = h5py.File(train_file, 'r')
    smiles_matrix = train_hdf5['smiles_matrix']
    classes = train_hdf5['classes']
    if path.isfile(model_file):
        print('Loading existing model ' + model_file)
        model = models.load_model(model_file, custom_objects={'sampling':cnn.Sampler().sampling})
    else:
        print('Creating new model')
        model = cnn.create_model_simple(smiles_matrix.shape[1:], classes.shape[1])
    checkpointer = ModelCheckpoint(filepath=model_file, verbose=0)
    reduce_learning_rate = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=2, min_lr=0.001)
    model_history = ModelHistory(model_file[:-3] + '-history.csv')
    print('Training model for ' + str(epochs) + ' epochs')
    model.fit(smiles_matrix, classes, epochs=epochs, shuffle='batch', batch_size=batch_size,
              callbacks=[checkpointer, reduce_learning_rate, model_history])
    train_hdf5.close()


class ModelHistory(Callback):

    def __init__(self, path):
        self._path_ = path

    def on_train_begin(self, logs=None):
        write_header = not path.isfile(self._path_)
        self._file_ = open(self._path_, 'a')
        if write_header:
            self._file_.write('loss,categorical_accuracy\n')

    def on_train_end(self, logs=None):
        self._file_.close()

    def on_epoch_end(self, epoch, logs=None):
        self._file_.write(str(logs.get('loss')) + ',' + str(logs.get('categorical_accuracy')) + '\n')
