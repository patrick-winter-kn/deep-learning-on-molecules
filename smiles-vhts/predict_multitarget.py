import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gc
import argparse
from util import predict_multitarget
import h5py
import re


def get_arguments():
    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('data', type=str, help='The source data')
    parser.add_argument('model_id', type=str, help='ID of the models')
    parser.add_argument('--batch_size', type=int, default=100, help='Size of the batches')
    return parser.parse_args()


args = get_arguments()
source_hdf5 = h5py.File(args.data, 'r')
ids = []
regex = re.compile('[0-9]+-classes')
for data_set in source_hdf5.keys():
    data_set = str(data_set)
    if regex.match(data_set):
        ids.append(data_set[:-8])
for ident in ids:
    predict_multitarget.predict(args.data, ident, args.model_id, args.batch_size)
source_hdf5.close()
gc.collect()
