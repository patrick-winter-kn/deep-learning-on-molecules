import gc
import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from util import learn


def get_arguments():
    parser = argparse.ArgumentParser(description='Train a model on the given data set')
    parser.add_argument('data', type=str, help='Input file containing the training data set')
    parser.add_argument('model', type=str, help='Model file (a new one will be created if it does not exist yet)')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs (default: 1)')
    parser.add_argument('--batch_size', type=int, default=100, help='Size of a batch (default: 100)')
    parser.add_argument('--validation', type=str, default=None, help='Validation data set used to generate statistics '
                                                                     'about the model(default: None)')
    return parser.parse_args()


args = get_arguments()
learn.train(args.data, args.validation, args.model, args.epochs, args.batch_size)
gc.collect()
