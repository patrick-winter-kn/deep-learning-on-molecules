import gc
import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from util import predict


def get_arguments():
    parser = argparse.ArgumentParser(description='Predict the activity for the given data')
    parser.add_argument('data', type=str, help='Input data file containing the smiles matrices')
    parser.add_argument('model', type=str, help='The model file')
    parser.add_argument('predictions', type=str, help='Output file that will contain the predictions')
    parser.add_argument('--batch_size', type=int, default=100, help='Size of the batches (default: 100)')
    return parser.parse_args()


args = get_arguments()
predict.predict(args.data, args.model, args.predictions, args.batch_size)
gc.collect()
