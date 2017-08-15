import h5py
import argparse
from progressbar import ProgressBar
import re

def get_arguments():
    parser = argparse.ArgumentParser(description='Analyze SMILES strings')
    parser.add_argument('file', type=str, help='The smiles file')
    return parser.parse_args()

args = get_arguments()
atom_pattern = re.compile('[a-z]')
branch_pattern = re.compile('[()]')
ring_pattern = re.compile('[0-9]+')
data_h5 = h5py.File(args.file, 'r')
smiles_data = data_h5['smiles']
occurences = {}
atoms = 0
chars = 0
branches = 0
rings = 0
with ProgressBar(max_value=len(smiles_data)) as progress:
    for i in range(len(smiles_data)):
        smiles = smiles_data[i].decode('utf-8')
        for j in range(len(smiles)):
            chars += 1
            char = smiles[j].lower()
            if atom_pattern.match(char):
                if char not in occurences:
                    occurences[char] = 0
                occurences[char] += 1
                atoms += 1
            if branch_pattern.match(char):
                branches += 1
            if ring_pattern.match(char):
                rings += 1
        progress.update(i+1)
cleaned_occurences = {}
for char in occurences:
    value = occurences[char]
    value /= atoms
    value = round(value,2)
    if value >= 0.01:
        cleaned_occurences[char] = value
branches /= chars
branches = round(branches,2)
rings /= chars
rings = round(rings, 2)
print(cleaned_occurences)
print('Branches: ' + str(branches))
print('Rings: ' + str(rings))
data_h5.close()
