import gc
import os
import re
import h5py
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from util import learn_multitarget


def get_arguments():
    parser = argparse.ArgumentParser(description='Train models for all targets including a shared feature generating '
                                                 'model')
    parser.add_argument('data', type=str, help='The source data file')
    parser.add_argument('model_id', type=str, help='ID of the models')
    parser.add_argument('--repeats', type=int, default=1, help='Number of repeated trainings over the different '
                                                               'targets (default: 1)')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs (default: 1)')
    parser.add_argument('--batch_size', type=int, default=100, help='Size of a batch (default: 100)')
    parser.add_argument('--validation', action='store_true', help='Use validation data set (default: False)')
    parser.add_argument('--freeze_features', action='store_true', help='Freeze the weights of the shared features '
                                                                       'model (default: False)')
    return parser.parse_args()


args = get_arguments()
source_hdf5 = h5py.File(args.data, 'r')
ids = []
regex = re.compile('[0-9]+-classes')
for data_set in source_hdf5.keys():
    data_set = str(data_set)
    if regex.match(data_set):
        ids.append(data_set[:-8])
for i in range(args.repeats):
    for ident in ids:
        learn_multitarget.train(args.data, ident, args.validation, args.batch_size, args.epochs, args.model_id,
                                args.freeze_features)
source_hdf5.close()
gc.collect()
