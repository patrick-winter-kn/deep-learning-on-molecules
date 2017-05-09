import argparse
from util import preprocess, partition, oversample


def get_arguments():
    parser = argparse.ArgumentParser(description='Prepares the data set for training')
    parser.add_argument('data', type=str, help='The data set containing the SMILES, class probability and partitioning')
    return parser.parse_args()


args = get_arguments()
prefix = args.data[:args.data.rfind('.')]
preprocess.preprocess(args.data, prefix + '-indices.h5', prefix + '-smiles_matrices.h5')
partition.write_partitions(args.data, prefix + '-smiles_matrices.h5', {1: 'train', 2: 'test', 3: 'validate'})
oversample.oversample(prefix + '-train.h5')
