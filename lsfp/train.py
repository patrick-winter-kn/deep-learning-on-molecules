import argparse
from util import operations


def get_arguments():
    parser = argparse.ArgumentParser(description='Train a model (continues training if the model already exists)')
    parser.add_argument('directory', type=str, help='Directory containing the prepared data')
    parser.add_argument('name', type=str, help='Name of the data set (name of the original file without path and file '
                                               'extension)')
    parser.add_argument('--id', type=str, default=None, help='Id of the model in case there are multiple for the same '
                                                             'data set')
    parser.add_argument('--batch_size', type=int, default=100, help='Number of data points that will be processed at '
                                                                    'once')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    return parser.parse_args()


args = get_arguments()
operations.train_model(args.directory, args.name, args.id, args.epochs, args.batch_size)
