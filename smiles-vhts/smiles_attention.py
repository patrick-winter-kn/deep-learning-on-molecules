import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gc
import argparse
from keras import models, activations
from vis.utils import utils
from vis import visualization
import h5py
from util import render_smiles


def get_arguments():
    parser = argparse.ArgumentParser(description='Generate attention map for SMILES based predictions')
    parser.add_argument('model', type=str, help='The model')
    parser.add_argument('data', type=str, help='The input data')
    parser.add_argument('index', type=int, help='Index of the data point to visualize')
    return parser.parse_args()


args = get_arguments()
model = models.load_model(args.model)
out_layer_index = utils.find_layer_idx(model, 'output')
model.layers[out_layer_index].activation = activations.linear
model = utils.apply_modifications(model)

data_h5 = h5py.File(args.data, 'r')
smiles_matrices_h5 = h5py.File(args.data[:-3] + '-smiles_matrices.h5', 'r')
smiles = data_h5['smiles'][args.index].decode('utf-8')
smiles_matrix = smiles_matrices_h5['smiles_matrix'][args.index]

heatmap = visualization.visualize_saliency(model, out_layer_index, filter_indices=[0], seed_input=smiles_matrix)

print('SMILES: ' + smiles)

render_smiles.render(smiles, '/home/winter/smiles-test.svg', 5, heatmap)

data_h5.close()
smiles_matrices_h5.close()

gc.collect()
