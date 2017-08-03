import h5py
import argparse
import os
from rdkit import Chem
from rdkit.Chem import Draw
from progressbar import ProgressBar
from util import partition_ref, oversample_ref, shuffle


def get_arguments():
    parser = argparse.ArgumentParser(description='Preprocess data for image based learning')
    parser.add_argument('data', type=str, help='Input data file containing the smiles, the classes and the partitions')
    return parser.parse_args()


args = get_arguments()
data_h5 = h5py.File(args.data, 'r')
smiles = data_h5['smiles']
prefix = args.data[:args.data.rfind('.')]
image_dir = args.data[:args.data.rfind('/')] + '/images/'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

drawingSize = (800, 800)
print('Generating images')
with ProgressBar(max_value=len(smiles)) as progress:
    for i in range(len(smiles)):
        image_path = image_dir + str(i) + '.png'
        if not os.path.exists(image_path):
            molecule = Chem.MolFromSmiles(smiles[i])
            Draw.MolToFile(molecule, image_path, size=drawingSize)
        progress.update(i+1)

if not os.path.exists(prefix + '-train.h5') or not os.path.exists(prefix + '-validate.h5'):
    partition_ref.write_partitions(args.data, {1: 'train', 2: 'validate'})
    oversample_ref.oversample(prefix + '-train.h5', args.data)
    shuffle.shuffle(prefix + '-train.h5')

data_h5.close()
