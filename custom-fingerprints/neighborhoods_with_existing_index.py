import argparse
from neighborhoods import neighborhoods


def get_arguments():
    parser = argparse.ArgumentParser(description='Create neighborhood fingerprints')
    parser.add_argument('smiles_file_path', type=str, help='Path to a HDF5 file containing a smiles column')
    parser.add_argument('index_file_path', type=str, help='Path to a HDF5 file containing a previously created index')
    return parser.parse_args()


args = get_arguments()
neighborhoods.fingerprint_with_existing_index(args.smiles_file_path, args.index_file_path)
