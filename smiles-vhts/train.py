import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gc
import argparse
from util import learn


def get_arguments():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('data', type=str, help='Training data set')
    parser.add_argument('model', type=str, help='Model')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=100, help='Size of a batch')
    return parser.parse_args()


args = get_arguments()
learn.train(args.data, args.model, 292, args.epochs, args.batch_size)
gc.collect()
