import gc
import os
import h5py
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
import numpy
from progressbar import ProgressBar
from data_structures import reference_data_set
from keras.callbacks import TensorBoard, ModelCheckpoint
from util.learn import  DrugDiscoveryEval
from util import actives_counter
from keras import models


def get_arguments():
    parser = argparse.ArgumentParser(description='Train model for images')
    parser.add_argument('data', type=str, help='The source data file')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs (default: 1)')
    parser.add_argument('--batch_size', type=int, default=50, help='Size of a batch (default: 50)')
    parser.add_argument('--validation', action='store_true', help='Use validation data set (default: False)')
    return parser.parse_args()


args = get_arguments()
image_dir = args.data[:args.data.rfind('/')] + '/images/'
data_h5 = h5py.File(args.data, 'r')
train_h5 = h5py.File(args.data[:args.data.rfind('.')] + '-train.h5', 'r')
train = train_h5['ref']
classes = reference_data_set.ReferenceDataSet(train_h5['ref'], data_h5['classes'])
# load one image to get dimensions
width, height = image.load_img(image_dir + str(train[0]) + '.png').size
img_array = numpy.zeros((len(train), width, height))
with ProgressBar(max_value=len(train)) as progress:
    for i in range(len(train)):
        img = image.load_img(image_dir + str(train[i]) + '.png')
        img_array[i] = image.img_to_array(img)
if os.path.exists(args.model):
    model = models.load_model(args.model)
else:
    model = InceptionV3(weights=None, input_shape=(width, height), classes=2)
checkpointer = ModelCheckpoint(filepath=args.model)
tensorboard = TensorBoard(log_dir=args.data[:args.data.rfind('.')] + '-tensorboard', histogram_freq=1, write_graph=True,
                          write_images=False, embeddings_freq=1)
callbacks = [tensorboard, checkpointer]
if args.validation:
    test_h5 = h5py.File(args.data[:args.data.rfind('.')] + '-validate.h5', 'r')
    test = test_h5['ref']
    test_classes = reference_data_set.ReferenceDataSet(test_h5['ref'], data_h5['classes'])
    actives = actives_counter.count(test_classes)
    test_img_array = numpy.zeros((len(test), width, height))
    with ProgressBar(max_value=len(test)) as progress:
        for i in range(len(test)):
            img = image.load_img(image_dir + str(test[i]) + '.png')
            test_img_array[i] = image.img_to_array(img)
    callbacks = [DrugDiscoveryEval([5, 10], (test_img_array, test_classes), args.batch_size, actives)] + callbacks
model.fit(img_array, classes, epochs=args.epochs, shuffle='batch', batch_size=args.batch_size, callbacks=callbacks)

data_h5.close()
train_h5.close()
if args.validation:
    test_h5.close()
gc.collect()
