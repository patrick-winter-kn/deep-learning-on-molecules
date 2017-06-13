import argparse
from util import random_forest
import h5py
from data_structures import reference_data_set


def get_arguments():
    parser = argparse.ArgumentParser(description='Predict with a random forest')
    parser.add_argument('data', type=str, help='The source data file')
    parser.add_argument('features', type=str, help='File containing the features data set')
    parser.add_argument('model', type=str, help='Model file')
    parser.add_argument('out_data', type=str, help='File containing the predictions')
    return parser.parse_args()


args = get_arguments()
prefix = args.data[:args.data.rfind('.')]
test_h5 = h5py.File(prefix + '-test.h5', 'r')
features_h5 = h5py.File(args.features, 'r')
features = reference_data_set.ReferenceDataSet(test_h5['ref'], features_h5['features'])
predictions_h5 = h5py.File(args.out_data, 'w')
predictions = random_forest.train(features, args.model)
predictions_h5.create_dataset('predictions', data=predictions)
predictions_h5.close()
test_h5.close()
features_h5.close()
