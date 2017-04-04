import argparse
from keras import models
import h5py


def get_arguments():
    parser = argparse.ArgumentParser(description='Train CNN on given fingerprints')
    parser.add_argument('data', type=str, help='Path to a HDF5 file containing the fingerprints')
    parser.add_argument('model', type=str, help='Path to the model.')
    return parser.parse_args()


args = get_arguments()
prefix = args.data[:args.data.rfind('.')]
model = models.load_model(args.model)
data_file = h5py.File(args.data)
prediction_file = h5py.File(prefix + '-prediction.h5', 'w')
data = data_file['fingerprint']
classes = model.predict(data)
prediction_file.create_dataset('class', data=classes)
data_file.close()
prediction_file.close()