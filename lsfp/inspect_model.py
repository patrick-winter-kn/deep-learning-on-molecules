import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import gc
from models import cnn


def get_arguments():
    parser = argparse.ArgumentParser(description='Inspects the CNN model for a given input size')
    parser.add_argument('input_size', type=int, help='Number of input features')
    return parser.parse_args()


args = get_arguments()
model = cnn.create_model(args.input_size, 2)
cnn.print_model(model)
