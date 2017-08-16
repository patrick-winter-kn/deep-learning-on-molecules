import h5py
import argparse
import random
import re
from progressbar import ProgressBar


atoms = {'o': 0.13, 's': 0.03, 'n': 0.12, 'c': 0.72}
branches = 0.19
rings = 0.14
atom_pattern = re.compile('[a-z]')


ring_label = 1


def get_arguments():
    parser = argparse.ArgumentParser(description='Generate SMILES like strings')
    parser.add_argument('file', type=str, help='The output file')
    parser.add_argument('number', type=int, help='Number of generated strings')
    parser.add_argument('--max_length', type=int, default=120, help='Maximum number of characters (default: 120)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    return parser.parse_args()


def is_branch(action, current_length, target_length):
    return action <= branches and current_length > 0 and current_length + 4 < target_length


def is_ring(action, current_length, target_length, label_size):
    if label_size > 1:
        label_size += 1
    return action > branches and action <= rings + branches and current_length > 0 and current_length + label_size * 2 + 2 < target_length


def pick_atom():
    rest = random.uniform(0.0, 1.0)
    for atom in atoms:
        rest -= atoms[atom]
        if rest <= 0:
            return atom


def is_atom(char):
    global atom_pattern
    return atom_pattern.match(char)


def generate_smiles(min_length, max_length, is_in_ring):
    global ring_label
    length = random.randint(min_length,max_length)
    string = ''
    while len(string) < length:
        action = random.uniform(0.0, 1.0)
        if is_branch(action, len(string), length) and is_atom(string[-1]):
            string += '(' + generate_smiles(1, length - len(string) - 3, False) + ')' + pick_atom()
        elif is_ring(action, len(string), length, len(str(ring_label))) and is_atom(string[-1]) and not is_in_ring:
            label = str(ring_label)
            if ring_label > 9:
                label = '%' + label
            ring_label += 1
            string += label + generate_smiles(2, length - len(string) - (len(label) * 2), True) + label
        else:
            string += pick_atom()
    return string


args = get_arguments()
random.seed(args.seed)
data_h5 = h5py.File(args.file, 'w')
smiles_data = data_h5.create_dataset('smiles', (args.number,), 'S' + str(args.max_length))
with ProgressBar(max_value=args.number) as progress:
    for i in range(args.number):
        ring_label = 1
        smiles_data[i] = generate_smiles(1, args.max_length, False).encode()
        progress.update(i+1)
data_h5.close()
