from os import path
import math
import numpy
import h5py
from keras.callbacks import Callback, TensorBoard
from data_structures import reference_data_set
from models import cnn_shared
from util.learn import DrugDiscoveryEval


def train(data_file, identifier, use_validation, batch_size, epochs, model_id, freeze_features):
    prefix = data_file[:data_file.rfind('.')]
    classes_hdf5 = h5py.File(data_file, 'r')
    smiles_path = prefix + '-smiles_matrices.h5'
    smiles_hdf5 = h5py.File(smiles_path, 'r')
    train_path = prefix + '-' + identifier + '-train.h5'
    train_hdf5 = h5py.File(train_path, 'r')
    if model_id:
        model_path = prefix + '-' + identifier + '-model-' + model_id + '.h5'
        feature_model_path = prefix + '-model-' + model_id + '.h5'
    else:
        model_path = prefix + '-' + identifier + '-model.h5'
        feature_model_path = prefix + '-model.h5'
    smiles_matrix = reference_data_set.ReferenceDataSet(train_hdf5['ref'], smiles_hdf5['smiles_matrix'])
    classes = reference_data_set.ReferenceDataSet(train_hdf5['ref'], classes_hdf5[identifier + '-classes'])
    val_data = None
    actives = None
    if use_validation:
        validation_path = prefix + '-' + identifier + '-validate.h5'
        validation_hdf5 = h5py.File(validation_path, 'r')
        val_smiles_matrix = reference_data_set.ReferenceDataSet(validation_hdf5['ref'], smiles_hdf5['smiles_matrix'])
        val_classes = reference_data_set.ReferenceDataSet(validation_hdf5['ref'], classes_hdf5[identifier + '-classes'])
        val_data = (val_smiles_matrix, val_classes)
        if 'actives' in validation_hdf5.attrs:
            actives = validation_hdf5.attrs['actives']
    model = cnn_shared.SharedFeaturesModel(smiles_matrix.shape[1:], classes.shape[1], not freeze_features)
    if path.isfile(model_path):
        model.load_predictions_model(model_path)
    if path.isfile(feature_model_path):
        model.load_features_model(feature_model_path)
    history_file = model_path[:-3] + '-history.csv'
    epoch = 0;
    if path.isfile(history_file):
        epoch = sum(1 for line in open(history_file)) - 1
    model_history = ModelHistory(history_file)
    tensorboard = TensorBoard(log_dir=model_path[:-3] + '-tensorboard', histogram_freq=1, write_graph=True,
                              write_images=False, embeddings_freq=1)
    callbacks = [model_history, tensorboard]
    if use_validation:
        callbacks = [DrugDiscoveryEval([5, 10], val_data, batch_size, actives)] + callbacks
    model.predictions_model.fit(smiles_matrix, classes, epochs=epochs, shuffle='batch', batch_size=batch_size,
                                initial_epoch=epoch,
                                callbacks=callbacks)
    model.save_predictions_model(model_path)
    model.save_features_model(feature_model_path)
    classes_hdf5.close()
    smiles_hdf5.close()
    train_hdf5.close()
    if use_validation:
        validation_hdf5.close()


class ModelHistory(Callback):

    def __init__(self, file_path):
        super().__init__()
        self._file_ = None
        self._path_ = file_path
        self.log_keys = None
        self.write_header = False

    def on_train_begin(self, logs=None):
        self.write_header = not path.isfile(self._path_)
        self._file_ = open(self._path_, 'a')

    def on_train_end(self, logs=None):
        self._file_.close()

    def on_epoch_end(self, epoch, logs=None):
        if self.log_keys is None:
            self.log_keys = sorted(list(logs.keys()))
            if self.write_header:
                self._file_.write(','.join(self.log_keys) + '\n')
        line = ''
        for key in self.log_keys:
            line += str(logs[key]) + ','
        self._file_.write(line[:-1] + '\n')
