import gc
import os
import h5py
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.preprocessing import image
from keras import models
from progressbar import ProgressBar
from data_structures import reference_data_set
import numpy
import math
from util import enrichment_plotter


def get_arguments():
    parser = argparse.ArgumentParser(description='Predict with model for images')
    parser.add_argument('data', type=str, help='The source data file')
    parser.add_argument('model', type=str, help='The model file to be used')
    parser.add_argument('--batch_size', type=int, default=5, help='Size of a batch (default: 5)')
    return parser.parse_args()


args = get_arguments()
image_dir = args.data[:args.data.rfind('/')] + '/images/'
data_h5 = h5py.File(args.data, 'r')
test_h5 = h5py.File(args.data[:args.data.rfind('.')] + '-validate.h5', 'r')
test = test_h5['ref']
classes = reference_data_set.ReferenceDataSet(test, data_h5['classes'])
# load one image to get dimensions
width, height = image.load_img(image_dir + str(test[0]) + '.png').size
model = models.load_model(args.model)
predictions_h5 = h5py.File(args.data[:args.data.rfind('.')] + '-validate-predictions.h5', 'w')
predictions = predictions_h5.create_dataset('predictions', (classes.shape[0], classes.shape[1]))
with ProgressBar(max_value=len(test)) as progress:
    for i in range(int(math.ceil(test.shape[0] / args.batch_size))):
        start = i * args.batch_size
        end = min(test.shape[0], (i + 1) * args.batch_size)
        img_array = numpy.zeros((end-start, width, height, 3), dtype=numpy.uint8)
        for j in range(end-start):
            img = image.load_img(image_dir + str(test[start+j]) + '.png')
            img_array[j] = image.img_to_array(img)
        results = model.predict(img_array)
        predictions[start:end] = results[:]
        progress.update(end)
predictions_h5.create_dataset('classes', data=classes)

enrichment_plotter.plot([predictions], ['images'], classes, [5, 10], args.data[:args.data.rfind('.')] + '-validate-plot.svg')

data_h5.close()
test_h5.close()
predictions_h5.close()
gc.collect()
