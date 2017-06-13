import argparse
from util import random_forest
import h5py
from data_structures import reference_data_set


def get_arguments():
    parser = argparse.ArgumentParser(description='Train a random forest based on previously generated features')
    parser.add_argument('data', type=str, help='The source data file')
    parser.add_argument('features', type=str, help='File containing the features data set')
    parser.add_argument('model', type=str, help='Model file')
    parser.add_argument('--nr_trees', type=int, default=100, help='Number of trees (default: 100)')
    return parser.parse_args()


args = get_arguments()
prefix = args.data[:args.data.rfind('.')]
classes_h5 = h5py.File(args.data, 'r')
train_h5 = h5py.File(prefix + '-train.h5', 'r')
classes = reference_data_set.ReferenceDataSet(train_h5['ref'], classes_h5['classes'])
features_h5 = h5py.File(args.features, 'r')
features = reference_data_set.ReferenceDataSet(train_h5['ref'], features_h5['features'])
random_forest.train(features, classes, args.model, args.nr_trees)
classes_h5.close()
train_h5.close()
features_h5.close()
