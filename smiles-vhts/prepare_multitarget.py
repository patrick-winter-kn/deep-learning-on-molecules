import re
import h5py
import argparse
from util import preprocess, partition_ref, oversample_ref, shuffle, actives_counter
from data_structures import reference_data_set


def get_arguments():
    parser = argparse.ArgumentParser(description='Prepares the given data set for training')
    parser.add_argument('data', type=str, help='The data set containing the SMILES, classes and partitions')
    parser.add_argument('--oversample', action='store_true',
                        help='Oversample underrepresented classes in the training dataset (default: False)')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the training data sets (default: False)')
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
if args.shuffle:
    for ident in ids:
        shuffle.shuffle(prefix + '-' + ident + '-train.h5')
classes_h5 = h5py.File(args.data)
for ident in ids:
    val_h5 = h5py.File(prefix + '-' + ident + '-validate.h5', 'a')
    val_classes = reference_data_set.ReferenceDataSet(val_h5['ref'], classes_h5[ident + '-classes'])
    print(ident)
    val_h5.attrs['actives'] = actives_counter.count(val_classes)
    val_h5.close()
classes_h5.close()
