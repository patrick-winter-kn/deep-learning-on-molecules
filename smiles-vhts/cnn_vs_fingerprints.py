import gc
import os
import argparse
import h5py
import re
from progressbar import ProgressBar
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from util import preprocess, fingerprints


def get_arguments():
    parser = argparse.ArgumentParser(description='Run the CNN vs fingerprints experiment')
    parser.add_argument('data', type=str, help='Input file containing the source data')
    return parser.parse_args()


args = get_arguments()
prefix = args.data[:args.data.rfind('.')]
# get IDs
ids = []
source_hdf5 = h5py.File(args.data, 'r')
regex = re.compile('[0-9]+-classes')
for data_set in source_hdf5.keys():
    data_set = str(data_set)
    if regex.match(data_set):
        ids.append(data_set[:-8])
source_hdf5.close()
# preprocessing (general)
indices_file = prefix + '-indices.h5'
matrices_file = prefix + '-smiles_matrices.h5'
fingerprints_file = prefix + '-fingerprints.h5'
if not os.path.isfile(indices_file) or not os.path.isfile(matrices_file):
    preprocess.preprocess(args.data, indices_file, matrices_file)
if not os.path.isfile(fingerprints_file):
    fingerprints.write_fingerprints(args.data, 'smiles', fingerprints_file, 'fingerprint', 1024)
with ProgressBar(max_value=len(ids)) as progress:
    i = 0
    for ident in ids:
        # TODO preprocessing (for x)
        # TODO CNN training
        # TODO CNN feature generation
        # TODO RF training (CNN)
        # TODO RF training (fingerprints)
        # TODO prediction (CNN)
        # TODO prediction (fingerprints)
        i += 1
        progress.update(i)
# TODO evaluation
gc.collect()
