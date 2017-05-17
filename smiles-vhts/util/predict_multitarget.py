import h5py
from keras import models
from progressbar import ProgressBar
import math
from data_structures import reference_data_set


def predict(data_file, identifier, model_id, batch_size):
    print('Predicting ' + identifier)
    prefix = data_file[:data_file.rfind('.')]
    test_file = prefix + '-' + identifier + '-test.h5'
    model_file = prefix + '-' + identifier + '-model-' + model_id + '.h5'
    predictions_file = prefix + '-' + identifier + '-predictions' + model_id + '.h5'
    smiles_file = prefix + '-smiles_matrices.h5'
    smiles_hdf5 = h5py.File(smiles_file, 'r')
    test_hdf5 = h5py.File(test_file, 'r')
    predictions_hdf5 = h5py.File(predictions_file, 'w')
    model = models.load_model(model_file)
    smiles_matrix = reference_data_set.ReferenceDataSet(test_hdf5['ref'], smiles_hdf5['smiles_matrix'])
    predictions = predictions_hdf5.create_dataset('predictions', (smiles_matrix.shape[0], model.output_shape[1]))
    with ProgressBar(max_value=len(smiles_matrix)) as progress:
        for i in range(math.ceil(smiles_matrix.shape[0]/batch_size)):
            start = i * batch_size
            end = min(smiles_matrix.shape[0], (i + 1) * batch_size)
            results = model.predict(smiles_matrix[start:end])
            predictions[start:end] = results[:]
            progress.update(end)
    smiles_hdf5.close()
    test_hdf5.close()
    predictions_hdf5.close()
