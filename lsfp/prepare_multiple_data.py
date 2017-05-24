import argparse
from util import operations
from os import path
import h5py, re


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
#ids = []
#source_hdf5 = h5py.File(args.data, 'r')
#regex = re.compile('[0-9]+-classes')
#for data_set in source_hdf5.keys():
#    data_set = str(data_set)
#    if regex.match(data_set):
#        ids.append(data_set[:-8])
#source_hdf5.close()
#for ident in ids:
operations.prepare_data(directory, name, args.radius, args.random)
