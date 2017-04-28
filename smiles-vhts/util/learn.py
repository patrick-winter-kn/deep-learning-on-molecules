from models import cnn, mlp
import h5py
from os import path
from keras import models
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback, TensorBoard
import numpy


def train(train_file, validation_file, model_file, epochs, batch_size):
    val = validation_file is not None
    train_hdf5 = h5py.File(train_file, 'r')
    smiles_matrix = train_hdf5['smiles_matrix']
    classes = train_hdf5['classes']
    monitor_metric = 'categorical_accuracy'
    val_data = None
    if val:
        val_hdf5 = h5py.File(validation_file, 'r')
        val_smiles_matrix = val_hdf5['smiles_matrix']
        val_classes = val_hdf5['classes']
        monitor_metric = 'val_categorical_accuracy'
        val_data = (val_smiles_matrix, val_classes)
    if path.isfile(model_file):
        print('Loading existing model ' + model_file)
        model = models.load_model(model_file)
    else:
        print('Creating new model')
        model = cnn.create_model(smiles_matrix.shape[1:], classes.shape[1])
    checkpointer = ModelCheckpoint(filepath=model_file, monitor=monitor_metric, save_best_only=True)
    reduce_learning_rate = ReduceLROnPlateau(monitor=monitor_metric, factor=0.2, patience=2, min_lr=0.001)
    tensorboard = TensorBoard(log_dir=model_file[:-3] + '-tensorboard', histogram_freq=1, write_graph=True,
                              write_images=False, embeddings_freq=1)
    model_history = ModelHistory(model_file[:-3] + '-history.csv', val, monitor_metric)
    print('Training model for ' + str(epochs) + ' epochs')
    model.fit(smiles_matrix, classes, epochs=epochs, shuffle='batch', batch_size=batch_size,
              callbacks=[checkpointer, reduce_learning_rate, tensorboard, model_history], validation_data=val_data)
    train_hdf5.close()


class ModelHistory(Callback):

    def __init__(self, file_path, with_validation, monitor):
        super().__init__()
        self._file_ = None
        self._path_ = file_path
        self._validation_ = with_validation
        self._monitor_ = monitor
        if 'loss' not in monitor:
            self._best_ = float('-inf')
            self._compare_ = numpy.greater
        else:
            self._best_ = float('inf')
            self._compare_ = numpy.less
        self._buffer_ = ''

    def on_train_begin(self, logs=None):
        write_header = not path.isfile(self._path_)
        self._file_ = open(self._path_, 'a')
        if write_header:
            self._file_.write('loss,categorical_accuracy')
            if self._validation_:
                self._file_.write(',val_loss,val_categorical_accuracy')
            self._file_.write('\n')

    def on_train_end(self, logs=None):
        self._file_.close()

    def on_epoch_end(self, epoch, logs=None):
        self._buffer_ += str(logs['loss']) + ',' + str(logs['categorical_accuracy'])
        if self._validation_:
            self._buffer_ += ',' + str(logs['val_loss']) + ',' + str(logs['val_categorical_accuracy'])
        self._buffer_ += '\n'
        if self._compare_(logs[self._monitor_], self._best_):
            self._best_ = logs[self._monitor_]
            self._file_.write(self._buffer_)
            self._buffer_ = ''
