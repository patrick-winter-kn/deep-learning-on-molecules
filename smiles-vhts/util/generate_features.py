import math
import h5py
from progressbar import ProgressBar
from keras import models


def generate_features(smiles_file, model_file, features_file, batch_size):
    print('Generating features')
    smiles_hdf5 = h5py.File(smiles_file, 'r')
    features_hdf5 = h5py.File(features_file, 'w')
    smiles_matrix = smiles_hdf5['smiles_matrix']
    model = models.load_model(model_file)
    features = features_hdf5.create_dataset('features', (smiles_matrix.shape[0], model.output_shape[1]))
    with ProgressBar(max_value=len(smiles_matrix)) as progress:
        for i in range(int(math.ceil(smiles_matrix.shape[0]/batch_size))):
            start = i * batch_size
            end = min(smiles_matrix.shape[0], (i + 1) * batch_size - 1)
            results = model.predict(smiles_matrix[start:end])
            features[start:end] = results[:]
            progress.update(end)
    smiles_hdf5.close()
    features_hdf5.close()
