from models import cnn, mlp
import h5py
from os import path
from keras import models
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback, TensorBoard
import numpy
import math


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
        monitor_metric = 'enrichment_auc'
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
    model_history = ModelHistory(model_file[:-3] + '-history.csv', monitor_metric)
    print('Training model for ' + str(epochs) + ' epochs')
    model.fit(smiles_matrix, classes, epochs=epochs, shuffle='batch', batch_size=batch_size,
              callbacks=[DrugDiscoveryEval([5,10]), checkpointer, reduce_learning_rate, tensorboard, model_history], validation_data=val_data)
    train_hdf5.close()


class ModelHistory(Callback):

    def __init__(self, file_path, monitor):
        super().__init__()
        self._file_ = None
        self._path_ = file_path
        self._monitor_ = monitor
        if 'loss' not in monitor:
            self._best_ = float('-inf')
            self._compare_ = numpy.greater
        else:
            self._best_ = float('inf')
            self._compare_ = numpy.less
        self._buffer_ = ''
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
        self._buffer_ += line[:-1] + '\n'
        if self._compare_(logs[self._monitor_], self._best_):
            self._best_ = logs[self._monitor_]
            self._file_.write(self._buffer_)
            self._buffer_ = ''


class DrugDiscoveryEval(Callback):

    def __init__(self, ef_percent):
        self.ef_percent = ef_percent
        self.positives = None

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(self.validation_data[0])
        # Get first column ([:,0], sort it (.argsort()) and reverse the order ([::-1]))
        indices = predictions[:,0].argsort()[::-1]
        # Start with a new line, don't print right of the progressbar
        print()
        for percent in self.ef_percent:
            ef = self.enrichment_factor(indices, percent)
            print('Enrichment Factor ' + str(percent) + '%: ' + str(ef))
            logs['enrichment_factor_' + str(percent)] = numpy.float64(ef)
        eauc = self.enrichment_auc(indices)
        print('Enrichment AUC: ' + str(eauc))
        logs['enrichment_auc'] = numpy.float64(eauc)
        dr = self.diversity_ratio(predictions)
        print('Diversity Ratio: ' + str(dr))
        logs['diversity_ratio'] = numpy.float64(dr)

    def positives_count(self):
        if self.positives is None:
            self.positives = 0
            for row in self.validation_data[1]:
                if numpy.where(row==max(row))[0]==0:
                    self.positives += 1
        return self.positives

    def enrichment_factor(self, indices, percent):
        ratio = percent * 0.01
        found = 0
        for i in range(math.floor(len(indices)*ratio)):
            row = self.validation_data[1][indices[i]]
            # Check if index (numpy.where) of maximum value (max(row)) in row is 0 (==0)
            # This means the active value is higher than the inactive value
            if numpy.where(row==max(row))[0]==0:
                found += 1
        return found / (self.positives_count() * ratio)

    def enrichment_auc(self, indices):
        found = 0
        sum = 0
        for i in range(len(indices)):
            row = self.validation_data[1][indices[i]]
            # Check if index (numpy.where) of maximum value (max(row)) in row is 0 (==0)
            # This means the active value is higher than the inactive value
            if numpy.where(row==max(row))[0]==0:
                found += 1
            sum += found
        # AUC = sum of found positives for every x / (positives * (number of samples + 1))
        # + 1 is added to the number of samples for the start with 0 samples selected
        return sum / (self.positives_count() * (len(self.validation_data[1]) + 1))

    def diversity_ratio(self, predictions):
        results = set()
        for i in range(len(predictions)):
            results.add(predictions[i][0])
        return len(results)/len(predictions)
