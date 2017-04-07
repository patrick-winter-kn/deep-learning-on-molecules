import argparse
from util import operations
from os import path


def get_arguments():
    parser = argparse.ArgumentParser(description='Prepare data for training')
    parser.add_argument('data', type=str, help='Path to a HDF5 file containing \'smiles\', \'classes\' and \'partition\'')
    parser.add_argument('--radius', type=int, default=2, help='Radius for the circular fingerprint')
    parser.add_argument('--random', type=bool, default=False, help='If the positions in the fingerprint should be '
                                                                   'assigned randomly or based on similarity')
    return parser.parse_args()


args = get_arguments()
path = path.abspath(args.data)
directory = path[:path.rfind('/')]
name = path[path.rfind('/') + 1 : path.rfind('.')]
operations.prepare_data(directory, name, args.radius, args.random)
