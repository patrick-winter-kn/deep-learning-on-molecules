import h5py
import argparse
import random


atoms = {'o': 0.12, 's': 0.02, 'n': 0.11, 'f': 0.01, 'c': 0.72, 'l': 0.01, 'h': 0.01}
branches = 0.19
rings = 0.14


ring_label = 1


def get_arguments():
    parser = argparse.ArgumentParser(description='Generate SMILES like strings')
    parser.add_argument('file', type=str, help='The output file')
    parser.add_argument('number', type=int, help='Number of generated strings')
    parser.add_argument('--max_length', type=int, default=120, help='Maximum number of characters (default: 120)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    return parser.parse_args()


def is_branch(action, current_length, target_length):
    return action <= branches and current_length > 0 and current_length + 3 < target_length


def is_ring(action, current_length, target_length, label_size):
    return action > branches and action <= rings + branches and current_length > 0 and current_length + label_size * 2 + 1 < target_length


def pick_atom():
    rest = random.uniform(0.0, 1.0)
    for atom in atoms:
        rest -= atoms[atom]
        if rest <= 0:
            return atom


def generate_smiles(max_length):
    global ring_label
    length = random.randint(1,max_length)
    string = ''
    while len(string) < length:
        action = random.uniform(0.0, 1.0)
        if is_branch(action, len(string), length):
            string += '(' + generate_smiles(length - len(string) - 3) + ')'
        elif is_ring(action, len(string), length, len(str(ring_label))):
            label = str(ring_label)
            ring_label += 1
            string += label + generate_smiles(length - len(string) - (len(label) * 2 + 1)) + label
        else:
            string += pick_atom()
    return string


args = get_arguments()
random.seed(args.seed)
data_h5 = h5py.File(args.file, 'w')
smiles_data = data_h5.create_dataset('smiles', (args.number,), 'S' + str(args.max_length))
for i in range(args.number):
    smiles_data[i] = generate_smiles(args.max_length).encode()
data_h5.close()
