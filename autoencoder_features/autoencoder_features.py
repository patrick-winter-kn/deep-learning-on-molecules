import gc
import os
import argparse
import h5py
import re
from progressbar import ProgressBar
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras import backend
from data_structures import reference_data_set
from util import preprocess, partition_ref, random_forest, enrichment_plotter
from autoencoder import trainer, predictor


def get_arguments():
    parser = argparse.ArgumentParser(description='Run the autoencoder as feature generator experiment')
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
results.write(',auc')
for enrichment_factor in enrichment_factors:
    results.write(',ef' + str(enrichment_factor))
results.write('\n')
if not os.path.isfile(indices_file) or not os.path.isfile(matrices_file):
    preprocess.preprocess(args.data, indices_file, matrices_file)
matrices_h5 = h5py.File(matrices_file, 'r')
# train autoencoder
model_file = prefix + '-autoencoder.h5'
trainer.train(model_file, matrices_h5['smiles_matrix'], 50, 2, 292)
# generate features
features_file = prefix + '-features.h5'
features_h5 = h5py.File(features_file, 'w')
predictor.encode(model_file, matrices_h5['smiles_matrix'], 292, features_h5)
with ProgressBar(max_value=len(ids)) as progress:
    i = 0
    for ident in ids:
        # preprocessing (for ident)
        train_file = prefix + '-' + ident + '-train.h5'
        validate_file = prefix + '-' + ident + '-validate.h5'
        if not os.path.isfile(train_file) or not os.path.isfile(validate_file):
            partition_ref.write_partitions(args.data, {1: 'train', 2: 'validate'}, ident)
        # RF training
        train_h5 = h5py.File(train_file, 'r')
        train_input = reference_data_set.ReferenceDataSet(train_h5['ref'], features_h5['latent_vectors'])
        train_output = reference_data_set.ReferenceDataSet(train_h5['ref'], source_h5[ident + '-classes'])
        train_output_classes = random_forest.numerical_to_classes(train_output)
        rf_file = prefix + '-' + ident + '-rf.h5'
        random_forest.train(train_input, train_output_classes, rf_file)
        # prediction
        test_h5 = h5py.File(validate_file, 'r')
        test_input = reference_data_set.ReferenceDataSet(test_h5['ref'], features_h5['latent_vectors'])
        predictions_file = prefix + '-' + ident + '-predictions.h5'
        predictions_h5 = h5py.File(predictions_file, 'w')
        predictions = random_forest.predict(test_input, rf_file)
        predictions_h5.create_dataset('predictions', data=predictions)
        # evaluation
        test_output = reference_data_set.ReferenceDataSet(test_h5['ref'], source_h5[ident + '-classes'])
        enrichment_plot_file = prefix + '-' + ident + '-plot.svg'
        aucs, efs = enrichment_plotter.plot(
            [predictions_h5['predictions']], ['Autoencoder'], test_output, enrichment_factors, enrichment_plot_file)
        results.write(str(ident))
        results.write(',' + str(aucs[0]))
        for ef in enrichment_factors:
            results.write(',' + str(efs[0][ef]))
        results.write('\n')
        # cleanup
        train_h5.close()
        test_h5.close()
        predictions_h5.close()
        backend.clear_session()
        # delete intermediate files to free space
        os.remove(predictions_file)
        # update progress
        i += 1
        progress.update(i)
source_h5.close()
matrices_h5.close()
features_h5.close()
results.close()
gc.collect()
