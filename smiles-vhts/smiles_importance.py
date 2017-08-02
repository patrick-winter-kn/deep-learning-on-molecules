import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gc
import argparse
from keras import models
import h5py
from util import render_smiles
import numpy


def get_arguments():
    parser = argparse.ArgumentParser(description='Generate importance map for SMILES based predictions')
    parser.add_argument('model', type=str, help='The model')
    parser.add_argument('data', type=str, help='The input data')
    parser.add_argument('output', type=str, help='The output SVG')
    parser.add_argument('index', type=int, help='Index of the data point to visualize')
    return parser.parse_args()


args = get_arguments()
model = models.load_model(args.model)

data_h5 = h5py.File(args.data, 'r')
smiles_matrices_h5 = h5py.File(args.data[:-3] + '-smiles_matrices.h5', 'r')
smiles = data_h5['smiles'][args.index].decode('utf-8')
smiles_matrix = smiles_matrices_h5['smiles_matrix'][args.index]
# Get activity for whole smiles_matrix
active = model.predict(numpy.array((smiles_matrix,)))[0][0]

differences = []
for i in range(len(smiles)):
    one_out_smiles_matrix = smiles_matrix.copy()
    # Set the one hot array at position i to all zeros
    one_out_smiles_matrix[i][numpy.nonzero(one_out_smiles_matrix[i])] = 0
    value = model.predict(numpy.array((one_out_smiles_matrix,)))[0][0]
    differences.append(active - value)

if max(differences) != 0:
    positive_factor = 1 / max(differences)
else:
    positive_factor = 0
if min(differences) != 0:
    negative_factor = 1 / abs(min(differences))
else:
    negative_factor = 0

print('Max: ' + str(max(differences)))
print('Positive factor: ' + str(positive_factor))
print('Min: ' + str(min(differences)))
print('Negative factor: ' + str(negative_factor))

heatmap = numpy.zeros((len(differences), 3), numpy.uint8)
for i in range(len(differences)):
    difference = differences[i]
    if difference > 0:
        heatmap[i][1] = 255 * difference * positive_factor
    else:
        heatmap[i][0] = 255 * abs(difference) * negative_factor

print('SMILES: ' + smiles)
print('p(active): ' + str(active))
print('Max red: ' + str(heatmap.max()))

render_smiles.render(smiles, args.output, 5, heatmap)

data_h5.close()
smiles_matrices_h5.close()

gc.collect()
