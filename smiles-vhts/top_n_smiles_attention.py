import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gc
import argparse
import h5py
from vis import visualization
from keras import models, activations
from vis.utils import utils
from util import render_smiles
from progressbar import ProgressBar


def get_arguments():
    parser = argparse.ArgumentParser(description='Generate attention map for SMILES based predictions')
    parser.add_argument('model', type=str, help='The model')
    parser.add_argument('data', type=str, help='The input data')
    parser.add_argument('assay', type=int, help='The assay ID')
    parser.add_argument('output', type=str, help='The output folder')
    parser.add_argument('n', type=int, default=None, help='The number of top SVGs to generate')
    parser.add_argument('--only_actives', type=bool, default=False,
                        help='Only render molecules that are actually active (default: False)')
    return parser.parse_args()


args = get_arguments()
model = models.load_model(args.model)
out_layer_index = utils.find_layer_idx(model, 'output')
model.layers[out_layer_index].activation = activations.linear
model = utils.apply_modifications(model)
data_h5 = h5py.File(args.data, 'r')
classes = data_h5[str(args.assay) + '-classes']
validate_h5 = h5py.File(args.data[:-3] + '-' + str(args.assay) + '-validate.h5', 'r')
prediction_h5 = h5py.File(args.data[:-3] + '-' + str(args.assay) + '-predictions.h5', 'r')
smiles_matrices_h5 = h5py.File(args.data[:-3] + '-smiles_matrices.h5', 'r')
predictions = prediction_h5['predictions']
indices = predictions[:, 0].argsort()[::-1]
if not os.path.exists(args.output):
    os.makedirs(args.output)
if args.n is None:
    count = len(indices)
else:
    count = min(args.n, len(indices))
j = 0
with ProgressBar(max_value=count) as progress:
    for i in range(count):
        index = validate_h5['ref'][indices[j]]
        while args.only_actives and classes[index][0] == 0:
            j += 1
            if j >= len(indices):
                break
            index = validate_h5['ref'][indices[j]]
        if j >= len(indices):
            break
        output = args.output + str(index) + '.svg'
        smiles = data_h5['smiles'][index].decode('utf-8')
        smiles_matrix = smiles_matrices_h5['smiles_matrix'][index]
        heatmap = visualization.visualize_saliency(model, out_layer_index, filter_indices=[0], seed_input=smiles_matrix)
        render_smiles.render(smiles, output, 5, heatmap)
        progress.update(i+1)
        j += 1
data_h5.close()
validate_h5.close()
prediction_h5.close()
gc.collect()
