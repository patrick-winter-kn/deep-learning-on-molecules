import argparse
from util import preprocess, partition, oversample, shuffle, actives_counter
import h5py


def get_arguments():
    parser = argparse.ArgumentParser(description='Prepares the given data set for training')
    parser.add_argument('data', type=str, help='The data set containing the SMILES, classes and partitions')
    parser.add_argument('--oversample', action='store_true',
                        help='Oversample underrepresented classes in the training dataset (default: False)')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the training data set (default: False)')
    return parser.parse_args()


args = get_arguments()
prefix = args.data[:args.data.rfind('.')]
preprocess.preprocess(args.data, prefix + '-indices.h5', prefix + '-smiles_matrices.h5')
partition.write_partitions(args.data, prefix + '-smiles_matrices.h5', {1: 'train', 2: 'validate'})
if args.oversample:
    oversample.oversample(prefix + '-train.h5')
if args.shuffle:
    shuffle.shuffle(prefix + '-train.h5')
val_h5 = h5py.File(prefix + '-validate.h5', 'a')
val_h5.attrs['actives'] = actives_counter.count(val_h5['classes'])
val_h5.close()
