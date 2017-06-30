import h5py
from data_structures import reference_data_set
from models import deep_mlp
from keras import models
import math
from progressbar import ProgressBar


def train(data_file, identifier, batch_size, epochs):
    prefix = data_file[:data_file.rfind('.')]
    classes_hdf5 = h5py.File(data_file, 'r')
    smiles_path = prefix + '-smiles_matrices.h5'
    smiles_hdf5 = h5py.File(smiles_path, 'r')
    train_path = prefix + '-' + identifier + '-train.h5'
    train_hdf5 = h5py.File(train_path, 'r')
    nn_model_path = prefix + '-' + identifier + '-nn.h5'
    smiles_matrix = reference_data_set.ReferenceDataSet(train_hdf5['ref'], smiles_hdf5['smiles_matrix'])
    classes = reference_data_set.ReferenceDataSet(train_hdf5['ref'], classes_hdf5[identifier + '-classes'])
    model = deep_mlp.create_model(smiles_matrix.shape[1:], classes.shape[1])
    model.fit(smiles_matrix, classes, epochs=epochs, shuffle='batch', batch_size=batch_size)
    model.save(nn_model_path)
    classes_hdf5.close()
    smiles_hdf5.close()
    train_hdf5.close()


def predict(smiles_matrix, model_file, predictions_file, batch_size):
    print('Predicting with NN')
    predictions_hdf5 = h5py.File(predictions_file, 'w')
    model = models.load_model(model_file)
    features = predictions_hdf5.create_dataset('predictions', (smiles_matrix.shape[0], model.output_shape[1]))
    with ProgressBar(max_value=len(smiles_matrix)) as progress:
        for i in range(int(math.ceil(smiles_matrix.shape[0]/batch_size))):
            start = i * batch_size
            end = min(smiles_matrix.shape[0], (i + 1) * batch_size)
            results = model.predict(smiles_matrix[start:end])
            features[start:end] = results[:]
            progress.update(end)
    predictions_hdf5.close()
