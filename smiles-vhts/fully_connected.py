import gc
import os
import argparse
import h5py
import re
from progressbar import ProgressBar
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from util import preprocess, partition_ref, oversample_ref, shuffle, learn_deep_mlp, enrichment_plotter
from keras import backend
from data_structures import reference_data_set


def get_arguments():
    parser = argparse.ArgumentParser(description='Run the fully connected experiment')
    parser.add_argument('data', type=str, help='Input file containing the source data')
    return parser.parse_args()


args = get_arguments()
enrichment_factors = [5, 10]
enrichment_factors = sorted(enrichment_factors)
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
results_file = prefix + '-results.csv'
results = open(results_file, 'w')
results.write(',auc-fc')
for enrichment_factor in enrichment_factors:
    results.write(',ef' + str(enrichment_factor) + '-fc')
results.write('\n')
if not os.path.isfile(indices_file) or not os.path.isfile(matrices_file):
    preprocess.preprocess(args.data, indices_file, matrices_file)
matrices_h5 = h5py.File(matrices_file, 'r')
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
        # NN training
        learn_deep_mlp.train(args.data, ident, 50, 2)
        # prediction (NN)
        test_h5 = h5py.File(validate_file, 'r')
        nn_predictions_file = prefix + '-' + ident + '-nn_predictions.h5'
        nn_model_file = prefix + '-' + ident + '-nn.h5'
        nn_test_input = reference_data_set.ReferenceDataSet(test_h5['ref'], matrices_h5['smiles_matrix'])
        learn_deep_mlp.predict(nn_test_input, nn_model_file, nn_predictions_file, 50)
        nn_predictions_h5 = h5py.File(nn_predictions_file, 'r')
        # evaluation
        test_output = reference_data_set.ReferenceDataSet(test_h5['ref'], source_h5[ident + '-classes'])
        enrichment_plot_file = prefix + '-' + ident + '-plot.svg'
        aucs, efs = enrichment_plotter.plot([nn_predictions_h5['predictions']], ['Fully Connected Layers'], test_output,
                                            enrichment_factors, enrichment_plot_file)
        results.write(str(ident))
        results.write(',' + str(aucs[0]))
        for ef in enrichment_factors:
            results.write(',' + str(efs[0][ef]))
        results.write('\n')
        # cleanup
        test_h5.close()
        nn_predictions_h5.close()
        backend.clear_session()
        # delete intermediate files to free space
        os.remove(nn_predictions_file)
        os.remove(cnn_features_file)
        # update progress
        i += 1
        progress.update(i)
source_h5.close()
matrices_h5.close()
results.close()
gc.collect()
