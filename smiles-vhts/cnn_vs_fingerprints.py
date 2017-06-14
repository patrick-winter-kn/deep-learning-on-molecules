import gc
import os
import argparse
import h5py
import re
from progressbar import ProgressBar
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from util import preprocess, fingerprints, partition_ref, oversample_ref, shuffle, learn_cnn, generate_features, random_forest
from keras import backend
from data_structures import reference_data_set


def get_arguments():
    parser = argparse.ArgumentParser(description='Run the CNN vs fingerprints experiment')
    parser.add_argument('data', type=str, help='Input file containing the source data')
    return parser.parse_args()


args = get_arguments()
prefix = args.data[:args.data.rfind('.')]
# get IDs
ids = []
source_h5 = h5py.File(args.data, 'r')
regex = re.compile('[0-9]+-classes')
for data_set in source_h5.keys():
    data_set = str(data_set)
    if regex.match(data_set):
        ids.append(data_set[:-8])
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
        # preprocessing (for ident)
        train_file = prefix + '-' + ident + '-train.h5'
        validate_file = prefix + '-' + ident + '-validate.h5'
        if not os.path.isfile(train_file) or not os.path.isfile(validate_file):
            partition_ref.write_partitions(args.data, {1: 'train', 2: 'validate'}, ident)
            oversample_ref.oversample(train_file, args.data, ident)
            shuffle.shuffle(train_file)
        # CNN training
        learn_cnn.train(args.data, ident, 50, 1)
        # CNN feature generation
        cnn_features_file = prefix + '-' + ident + '-cnn_features.h5'
        cnn_model_file = prefix + '-' + ident + '-cnn.h5'
        generate_features.generate_features(matrices_file, cnn_model_file, cnn_features_file, 50)
        # RF training (CNN)
        train_h5 = h5py.File(train_file, 'r')
        cnn_features_h5 = h5py.File(cnn_features_file, 'r')
        cnn_train_input = reference_data_set.ReferenceDataSet(train_h5['ref'], cnn_features_h5['features'])
        train_output = reference_data_set.ReferenceDataSet(train_h5['ref'], source_h5[ident + '-classes'])
        rf_cnn_file = prefix + '-' + ident + '-rf_cnn.h5'
        random_forest.train(cnn_train_input, train_output, rf_cnn_file)
        # RF training (fingerprints)
        fingerprints_h5 = h5py.File(fingerprints_file, 'r')
        fp_train_input = reference_data_set.ReferenceDataSet(train_h5['ref'], fingerprints_h5['features'])
        rf_fp_file = prefix + '-' + ident + '-rf_fp.h5'
        random_forest.train(fp_train_input, train_output, rf_fp_file)
        # prediction (CNN)
        test_h5 = h5py.File(validate_file, 'r')
        cnn_test_input = reference_data_set.ReferenceDataSet(test_h5['ref'], cnn_features_h5['features'])
        cnn_predictions_file = prefix + '-' + ident + '-predictions_cnn.h5'
        cnn_predictions_h5 = h5py.File(cnn_predictions_file, 'w')
        predictions = random_forest.predict(cnn_test_input, rf_cnn_file)
        cnn_predictions_h5.create_dataset('predictions', data=predictions)
        # prediction (fingerprints)
        fp_test_input = reference_data_set.ReferenceDataSet(test_h5['ref'], fingerprints_h5['features'])
        fp_predictions_file = prefix + '-' + ident + '-predictions_fp.h5'
        fp_predictions_h5 = h5py.File(fp_predictions_file, 'w')
        predictions = random_forest.predict(fp_test_input, rf_fp_file)
        fp_predictions_h5.create_dataset('predictions', data=predictions)
        train_h5.close()
        test_h5.close()
        cnn_features_h5.close()
        fingerprints_h5.close()
        cnn_predictions_h5.close()
        fp_predictions_h5.close()
        backend.clear_session()
        i += 1
        progress.update(i)
# TODO evaluation
source_h5.close()
gc.collect()