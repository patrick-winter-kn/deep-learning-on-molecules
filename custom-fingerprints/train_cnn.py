import argparse
import numpy
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
    parser.add_argument('--batch_size', type=int, default=50, help='Number of samples in a batch')
    return parser.parse_args()


def argmax(self, axis=None, out=None):
    # This is only implemented for axis=1
    if not hasattr(self, '_preprocessed_argmax_'):
        result = []
        for value in self:
            max_index = 0
            max_value = value[0]
            for i in range(1, len(value)):
                inner_value = value[i]
                if inner_value > max_value:
                    max_value = inner_value
                    max_index = i
            result.append(max_index)
        self._preprocessed_argmax_ = numpy.array(result)
    return self._preprocessed_argmax_


args = get_arguments()
data_file = h5py.File(args.data, 'r')
data = data_file['fingerprint']
classes = data_file['class']
type(classes).argmax = argmax
class_zero_sum = 0
class_one_sum = 0
for i in range(classes.shape[0]):
    class_zero_sum += classes[i][0]
    class_one_sum += classes[i][1]
class_weight = {0:classes.shape[0]/class_zero_sum,1:classes.shape[0]/class_one_sum}
if path.isfile(args.model):
    print('Loading existing model')
    model = models.load_model(args.model)
else:
    print('Creating new model')
    model = cnn.create_model(data.shape[1])
checkpointer = ModelCheckpoint(filepath=args.model, verbose=1)
reduce_learning_rate = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=2, min_lr=0.001)
print('Training model for ' + str(args.epochs) + ' epochs')
model.fit(data, classes, nb_epoch=args.epochs, shuffle='batch', batch_size=args.batch_size, class_weight=class_weight,
          callbacks=[checkpointer, reduce_learning_rate])
print('Training completed')
data_file.close()
