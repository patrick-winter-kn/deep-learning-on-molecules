import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gc
import argparse
from util import learn_multitarget
import h5py
import re


def get_arguments():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('data', type=str, help='Training data set')
    parser.add_argument('model_id', type=str, help='Model')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=100, help='Size of a batch')
    parser.add_argument('--validation', action='store_true', help='Use validation data set')
    return parser.parse_args()


args = get_arguments()
source_hdf5 = h5py.File(args.data, 'r')
ids = []
regex = re.compile('[0-9]+-classes')
for data_set in source_hdf5.keys():
    data_set = str(data_set)
    if regex.match(data_set):
        ids.append(data_set[:-8])
for i in range(args.epochs):
    for ident in ids:
        learn_multitarget.train(args.data, ident, args.validation, args.batch_size, args.model_id)
source_hdf5.close()
gc.collect()
