import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gc
import argparse
from util import predict


def get_arguments():
    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('data', type=str, help='The data to predict')
    parser.add_argument('model', type=str, help='The model')
    parser.add_argument('predictions', type=str, help='File containing the predictions')
    parser.add_argument('--batch_size', type=int, default=100, help='Size of the batches')
    return parser.parse_args()


args = get_arguments()
predict.predict(args.data, args.model, args.predictions, 292, args.batch_size)
gc.collect()
