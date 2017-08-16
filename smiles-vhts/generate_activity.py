import argparse
import h5py
from rdkit import Chem
from progressbar import ProgressBar


def get_arguments():
    parser = argparse.ArgumentParser(description='Generate activity based on SMILES that contain a given substructure')
    parser.add_argument('file', type=str, help='The SMILES file')
    parser.add_argument('substructure', type=str, help='The substructure')
    parser.add_argument('--dataset_name', type=str, default='classes', help='Name of the dataset holding the activity data (default: classes)')
    return parser.parse_args()


args = get_arguments()
substructure = Chem.MolFromSmiles(args.substructure, sanitize=False)
data_h5 = h5py.File(args.file, 'r+')
smiles_data = data_h5['smiles']
classes = data_h5.create_dataset(args.dataset_name, (smiles_data.shape[0], 2))
with ProgressBar(max_value=len(smiles_data)) as progress:
    for i in range(len(smiles_data)):
        smiles = smiles_data[i].decode('utf-8')
        molecule = Chem.MolFromSmiles(smiles, sanitize=False)
        active = molecule.HasSubstructMatch(substructure)
        if active:
            classes[i,0] = 1.0
            classes[i,1] = 0.0
        else:
            classes[i,0] = 0.0
            classes[i,1] = 1.0
        progress.update(i+1)
data_h5.close()
