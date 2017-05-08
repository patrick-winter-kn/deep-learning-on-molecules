import h5py
from models import cnn
from keras import models
from progressbar import ProgressBar
import math


def predict(test_file, model_file, predictions_file, batch_size):
    print('Predicting')
    test_hdf5 = h5py.File(test_file, 'r')
    predictions_hdf5 = h5py.File(predictions_file, 'w')
    model = models.load_model(model_file)
    smiles_matrix = test_hdf5['smiles_matrix']
    classes = test_hdf5['classes']
    predictions = predictions_hdf5.create_dataset('predictions', (classes.shape[0], classes.shape[1]))
    with ProgressBar(max_value=len(smiles_matrix)) as progress:
        for i in range(math.ceil(smiles_matrix.shape[0]/batch_size)):
            start = i * batch_size
            end = min(smiles_matrix.shape[0], (i + 1) * batch_size - 1)
            results = model.predict(smiles_matrix[start:end])
            predictions[start:end] = results[:]
            progress.update(end)
    predictions_hdf5.create_dataset('classes', data=classes)
    test_hdf5.close()
    predictions_hdf5.close()
