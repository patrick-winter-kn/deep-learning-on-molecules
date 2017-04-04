import argparse
from network import cnn
from os import path
from keras import models
import h5py
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


def get_arguments():
    parser = argparse.ArgumentParser(description='Train CNN on given fingerprints')
    parser.add_argument('data', type=str, help='Path to a HDF5 file containing the fingerprints')
    parser.add_argument('model', type=str, help='Path to the model.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=500, help='Number of samples in a batch')
    return parser.parse_args()


args = get_arguments()
data_file = h5py.File(args.data, 'r')
data = data_file['fingerprint']
classes = data_file['class']
if path.isfile(args.model):
    print('Loading existing model')
    model = models.load_model(args.model)
else:
    print('Creating new model')
    model = cnn.create_model(data.shape[1])
checkpointer = ModelCheckpoint(filepath=args.model, verbose=1)
reduce_learning_rate = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.001)
print('Training model for ' + str(args.epochs) + ' epochs')
model.fit(data, classes, nb_epoch=args.epochs, shuffle='batch', batch_size=args.batch_size, callbacks=[checkpointer,
                                                                                                       reduce_learning_rate])
print('Training completed')
data_file.close()
