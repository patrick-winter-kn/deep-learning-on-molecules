from os import path
import math
import numpy
import h5py
from keras.callbacks import Callback
from data_structures import reference_data_set
from models import cnn_shared


def train(data_file, identifier, use_validation, batch_size, epochs, model_id, freeze_features):
    prefix = data_file[:data_file.rfind('.')]
    classes_hdf5 = h5py.File(data_file, 'r')
    smiles_path = prefix + '-smiles_matrices.h5'
    smiles_hdf5 = h5py.File(smiles_path, 'r')
    train_path = prefix + '-' + identifier + '-train.h5'
    train_hdf5 = h5py.File(train_path, 'r')
    model_path = prefix + '-' + identifier + '-model-' + model_id + '.h5'
    feature_model_path = prefix + '-model-' + model_id + '.h5'
    smiles_matrix = reference_data_set.ReferenceDataSet(train_hdf5['ref'], smiles_hdf5['smiles_matrix'])
    classes = reference_data_set.ReferenceDataSet(train_hdf5['ref'], classes_hdf5[identifier + '-classes'])
    val_data = None
    if use_validation:
        validation_path = prefix + '-' + identifier + '-validate.h5'
        validation_hdf5 = h5py.File(validation_path, 'r')
        val_smiles_matrix = reference_data_set.ReferenceDataSet(validation_hdf5['ref'], smiles_hdf5['smiles_matrix'])
        val_classes = reference_data_set.ReferenceDataSet(validation_hdf5['ref'], classes_hdf5[identifier + '-classes'])
        val_data = (val_smiles_matrix, val_classes)
    model = cnn_shared.SharedFeaturesModel(smiles_matrix.shape[1:], classes.shape[1], not freeze_features)
    if path.isfile(model_path):
        model.load_predictions_model(model_path)
    if path.isfile(feature_model_path):
        model.load_features_model(feature_model_path)
    model_history = ModelHistory(model_path[:-3] + '-history.csv')
    model.predictions_model.fit(smiles_matrix, classes, epochs=epochs, shuffle='batch', batch_size=batch_size,
                                callbacks=[DrugDiscoveryEval([5, 10], val_data), model_history])
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


class DrugDiscoveryEval(Callback):

    def __init__(self, ef_percent, validation_data):
        super().__init__()
        self.ef_percent = ef_percent
        self.positives = None
        self.val_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        if self.val_data:
            # Start with a new line, don't print right of the progressbar
            print()
            print('Predicting with intermediate model...')
            predictions = self.model.predict(self.val_data[0])
            # Get first column ([:,0], sort it (.argsort()) and reverse the order ([::-1]))
            indices = predictions[:, 0].argsort()[::-1]
            efs, eauc = self.enrichment_stats(indices)
            for percent in efs.keys():
                print('Enrichment Factor ' + str(percent) + '%: ' + str(efs[percent]))
                logs['enrichment_factor_' + str(percent)] = numpy.float64(efs[percent])
            print('Enrichment AUC: ' + str(eauc))
            logs['enrichment_auc'] = numpy.float64(eauc)
            dr = self.diversity_ratio(predictions)
            print('Diversity Ratio: ' + str(dr))
            logs['diversity_ratio'] = numpy.float64(dr)

    def positives_count(self):
        if self.positives is None:
            self.positives = 0
            for row in self.val_data[1]:
                if numpy.where(row == max(row))[0] == 0:
                    self.positives += 1
        return self.positives

    def enrichment_stats(self, indices):
        # efs maps the percent to the number of found positives
        efs = {}
        for percent in self.ef_percent:
            efs[percent] = 0
        found = 0
        curve_sum = 0
        for i in range(len(indices)):
            row = self.val_data[1][indices[i]]
            # Check if index (numpy.where) of maximum value (max(row)) in row is 0 (==0)
            # This means the active value is higher than the inactive value
            if numpy.where(row == max(row))[0] == 0:
                found += 1
                for percent in efs.keys():
                    # If i is still part of the fraction count the number of founds up
                    if i < int(math.floor(len(indices)*(percent*0.01))):
                        efs[percent] += 1
            curve_sum += found
        # AUC = sum of found positives for every x / (positives * (number of samples + 1))
        # + 1 is added to the number of samples for the start with 0 samples selected
        auc = curve_sum / (self.positives_count() * (len(self.val_data[1]) + 1))
        # Turn number of found positives into enrichment factor by dividing the number of positives found at random
        for percent in efs.keys():
            efs[percent] /= (self.positives_count() * (percent * 0.01))
        return efs, auc

    @staticmethod
    def diversity_ratio(predictions):
        results = set()
        for i in range(len(predictions)):
            results.add(predictions[i][0])
        return len(results)/len(predictions)
