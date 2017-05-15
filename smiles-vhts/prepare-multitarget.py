import argparse
from util import preprocess, partition_ref, oversample_ref
import h5py
import re


def get_arguments():
    parser = argparse.ArgumentParser(description='Prepares the data set for training')
    parser.add_argument('data', type=str, help='The data set containing the SMILES, class probability and partitioning')
    parser.add_argument('--oversample', action='store_true',
                        help='Oversample underrepresented classes in the training dataset')
    return parser.parse_args()


args = get_arguments()
prefix = args.data[:args.data.rfind('.')]
preprocess.preprocess(args.data, prefix + '-indices.h5', prefix + '-smiles_matrices.h5')
ids = []
source_hdf5 = h5py.File(args.data, 'r')
regex = re.compile('[0-9]+-classes')
for data_set in source_hdf5.keys():
    data_set = str(data_set)
    if regex.match(data_set):
        ids.append(data_set[:-8])
source_hdf5.close()
for ident in ids:
    partition_ref.write_partitions(args.data, {1: 'train', 2: 'test', 3: 'validate'}, ident)
if args.oversample:
    for ident in ids:
        oversample_ref.oversample(prefix + '-' + ident + '-train.h5', args.data, ident)
