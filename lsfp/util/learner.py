from keras import models
from models import cnn
import h5py
import math
from progressbar import ProgressBar
from os import path
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy


def train_model(train_file, model_file, epochs, batch_size):
    print('Training model \'' + model_file + '\'')
    train_hdf5 = h5py.File(train_file, 'r')
    fingerprints = train_hdf5['fingerprint']
    classes = train_hdf5['classes']
    if path.isfile(model_file):
        print('Loading existing model')
        model = models.load_model(model_file)
    else:
        print('Creating new model')
        model = cnn.create_model(fingerprints.shape[1], classes.shape[1])
    # print('Model:')
    # cnn.print_model(model)
    type(classes).argmax = argmax
    class_zero_sum = 0
    class_one_sum = 0
    for i in range(classes.shape[0]):
        class_zero_sum += classes[i][0]
        class_one_sum += classes[i][1]
    class_weight = {0:classes.shape[0]/class_zero_sum,1:classes.shape[0]/class_one_sum}
    checkpointer = ModelCheckpoint(filepath=model_file, verbose=0)
    reduce_learning_rate = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=2, min_lr=0.001)
    print('Training model for ' + str(epochs) + ' epochs')
    model.fit(fingerprints, classes, epochs=epochs, shuffle='batch', batch_size=batch_size,
              class_weight=class_weight, callbacks=[checkpointer, reduce_learning_rate])
    print('Training completed')
    train_hdf5.close()



def predict(test_file, model_file, predictions_file, batch_size):
    print('Predicting')
    model = models.load_model(model_file)
    test_hdf5 = h5py.File(test_file, 'r')
    predictions_hdf5 = h5py.File(predictions_file, 'w')
    classes = test_hdf5['classes']
    fingerprints = test_hdf5['fingerprint']
    predictions = predictions_hdf5.create_dataset('predictions', (classes.shape[0], classes.shape[1]))
    with ProgressBar(max_value=fingerprints.shape[0]) as progress:
        for i in range(math.ceil(fingerprints.shape[0]/batch_size)):
            start = i * batch_size
            end = min(fingerprints.shape[0], (i + 1) * batch_size - 1)
            results = model.predict(fingerprints[start:end])
            predictions[start:end] = results[:]
            progress.update(end)
    predictions_hdf5.create_dataset('classes', data=classes)
    test_hdf5.close()
    predictions_hdf5.close()


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
