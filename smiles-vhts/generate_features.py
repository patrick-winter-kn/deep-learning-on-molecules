import gc
import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from util import generate_features


def get_arguments():
    parser = argparse.ArgumentParser(description='Generate features using a previously trained model')
    parser.add_argument('data', type=str, help='File containing the input smiles matrices')
    parser.add_argument('model', type=str, help='The model file')
    parser.add_argument('features', type=str, help='Output file that will contain the generated features')
    parser.add_argument('--batch_size', type=int, default=100, help='Size of the batches (default: 100)')
    return parser.parse_args()


args = get_arguments()
generate_features.generate_features(args.data, args.model, args.features, args.batch_size)
gc.collect()
