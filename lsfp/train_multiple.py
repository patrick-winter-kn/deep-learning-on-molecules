import gc
import os
import re
import h5py
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from util import learn_multitarget
from keras import backend
from progressbar import ProgressBar


def get_arguments():
    parser = argparse.ArgumentParser(description='Train models for all targets including a shared feature generating '
                                                 'model')
    parser.add_argument('data', type=str, help='The source data file')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs (default: 1)')
    parser.add_argument('--batch_size', type=int, default=50, help='Size of a batch (default: 100)')
    parser.add_argument('--validation', action='store_true', help='Use validation data set (default: False)')
    return parser.parse_args()


args = get_arguments()
source_hdf5 = h5py.File(args.data, 'r')
ids = []
regex = re.compile('[0-9]+-classes')
for data_set in source_hdf5.keys():
    data_set = str(data_set)
    if regex.match(data_set):
        ids.append(data_set[:-8])
with ProgressBar(max_value=len(ids)) as progress:
    i = 0
    for ident in ids:
        print('========== ' + ident + ' ==========')
        learn_multitarget.train(args.data, ident, args.validation, args.batch_size, args.epochs, 'sorted')
        backend.clear_session()
        learn_multitarget.train(args.data, ident, args.validation, args.batch_size, args.epochs, 'random')
        backend.clear_session()
        i += 1
        progress.update(i)
source_hdf5.close()
gc.collect()
