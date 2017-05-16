import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gc
import argparse
from util import generate_features


def get_arguments():
    parser = argparse.ArgumentParser(description='Generate features')
    parser.add_argument('data', type=str, help='The smiles matrices file')
    parser.add_argument('model', type=str, help='The features model')
    parser.add_argument('features', type=str, help='File containing the features')
    parser.add_argument('--batch_size', type=int, default=100, help='Size of the batches')
    return parser.parse_args()


args = get_arguments()
generate_features.generate_features(args.data, args.model, args.features, args.batch_size)
gc.collect()
