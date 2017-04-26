import argparse
from util import operations


def get_arguments():
    parser = argparse.ArgumentParser(description='Oversampling')
    parser.add_argument('directory', type=str, help='Directory')
    parser.add_argument('name', type=str, help='Data set name')
    return parser.parse_args()


args = get_arguments()
operations.oversample(args.directory, args.name)
