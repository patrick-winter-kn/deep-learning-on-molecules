from rdkit import Chem
from functools import total_ordering
import argparse
import h5py
from progressbar import ProgressBar
import numpy


def get_arguments():
    parser = argparse.ArgumentParser(description='Create neighborhood fingerprints')
    parser.add_argument('input_table', type=str, help='HDF5 file containing a smiles column')
    return parser.parse_args()


def main(input_table):
    prefix = input_table[:input_table.rfind('.')]
    neighborhoods_table = prefix + '-neighborhoods.h5'
    index_table = prefix + '-index.h5'
    fingerprint_table = prefix + '-fingerprint.h5'
    input_file = h5py.File(input_table, 'r')
    neighborhoods_file = h5py.File(neighborhoods_table, 'w')
    index_file = h5py.File(index_table, 'w')
    fingerprint_file = h5py.File(fingerprint_table, 'w')
    smiles = input_file['smiles']
    neighborhoods_out = neighborhoods_file.create_dataset('neighborhoods', (len(smiles),), 'S2000')
    all_neighborhoods = set()
    print('Extracting neighborhoods')
    with ProgressBar(max_value=len(smiles)) as progress:
        for i in range(len(smiles)):
            neighborhoods = neighborhoods_from_smiles(smiles[i])
            for neighborhood in neighborhoods:
                all_neighborhoods.add(neighborhood)
            neighborhoods_out[i] = neighborhoods_to_string(neighborhoods).encode('utf-8')
            progress.update(i)
    index_out = index_file.create_dataset('index', (len(all_neighborhoods),), 'S2000')
    i = 0
    print('Writing index positions')
    with ProgressBar(max_value=len(all_neighborhoods)) as progress:
        for neighborhood in sorted(all_neighborhoods):
            index_out[i] = str(neighborhood).encode('utf-8')
            i += 1
            progress.update(i)
    index_dict = dict_from_file(index_file)
    fingerprint_out = fingerprint_file.create_dataset('fingerprint', (neighborhoods_out.shape[0], index_out.shape[0]),
                                                      'I')
    print('Generating fingerprints')
    with ProgressBar(max_value=neighborhoods_out.shape[0]) as progress:
        for i in range(neighborhoods_out.shape[0]):
            array = numpy.zeros(index_out.shape[0], dtype=numpy.uint32)
            neighborhoods = neighborhoods_from_string(neighborhoods_out[i].decode('utf-8'))
            for neighborhood in neighborhoods:
                index = index_dict[neighborhood]
                array[index] = neighborhoods[neighborhood]
            fingerprint_out[i] = array
            progress.update(i+1)
    input_file.close()
    neighborhoods_file.close()
    index_file.close()
    fingerprint_file.close()


def dict_from_file(file):
    index = file['index']
    index_dict = {}
    print('Creating lookup table')
    with ProgressBar(max_value=index.shape[0]) as progress:
        for i in range(index.shape[0]):
            index_dict[Neighborhood.from_string(index[i].decode('utf-8'))] = i
            progress.update(i+1)
    return index_dict


def neighborhoods_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    neighborhoods = {}
    for atom in mol.GetAtoms():
        atom_neighborhood = Neighborhood.from_atom(atom, 2, set())
        if atom_neighborhood in neighborhoods:
            neighborhoods[atom_neighborhood] += 1
        else:
            neighborhoods[atom_neighborhood] = 1
    return neighborhoods


def neighborhoods_to_string(neighborhoods):
    string = '{'
    for key in sorted(neighborhoods):
        string += str(key) + ':' + str(neighborhoods[key]) + ','
    if string.endswith(','):
        string = string[:-1]
    string += '}'
    return string


def neighborhoods_from_string(string):
    neighborhoods = {}
    string = string[1:-1]
    while len(string) > 0:
        colon = string.find(':')
        key = Neighborhood.from_string(string[:colon])
        string = string[colon+1:]
        comma = string.find(',')
        if comma < 0:
            comma = len(string)
        value = int(string[:comma])
        neighborhoods[key] = value
        if comma < len(string):
            string = string[comma+1:]
        else:
            string = ''
    return neighborhoods


@total_ordering
class Neighborhood:

    def __init__(self, symbol, neighbors):
        self.symbol = symbol
        self.neighbors = sorted(neighbors)

    @classmethod
    def from_atom(cls, atom, radius, black_list):
        symbol = atom.GetSymbol()
        neighbors = []
        if radius > 0:
            neighbor_black_list = black_list.copy()
            neighbor_black_list.add(atom.GetIdx())
            for neighbor_atom in atom.GetNeighbors():
                if neighbor_atom.GetIdx() not in black_list:
                    neighbors.append(Neighborhood.from_atom(neighbor_atom, radius - 1, neighbor_black_list))
        return cls(symbol, neighbors)

    @classmethod
    def from_string(cls, string):
        if '[' not in string:
            return cls(string, [])
        else:
            symbol = string[:string.find('[')]
            string = string[string.find('[')+1:-1]
            neighbors = []
            neighbor_strings = []
            tmp_string = ''
            list_depth = 0
            for character in string:
                if character == ',' and list_depth < 1:
                    neighbor_strings.append(tmp_string)
                    tmp_string = ''
                else:
                    tmp_string += character
                    if character == '[':
                        list_depth += 1
                    elif character == ']':
                        list_depth -= 1
            if len(tmp_string) > 0:
                neighbor_strings.append(tmp_string)
            for neighbor_string in neighbor_strings:
                neighbors.append(Neighborhood.from_string(neighbor_string))
            return cls(symbol, neighbors)

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return str(self).__hash__()

    def __lt__(self, other):
        if self.symbol != other.symbol:
            if self.symbol < other.symbol:
                return True
            else:
                return False
        elif len(self.neighbors) != len(other.neighbors):
            if len(self.neighbors) < len(other.neighbors):
                return True
            else:
                return False
        else:
            for i in range(len(self.neighbors)):
                if self.neighbors[i] != other.neighbors[i]:
                    if self.neighbors[i] < other.neighbors[i]:
                        return True
                    else:
                        return False
        return False

    def __repr__(self):
        representation = self.symbol
        if len(self.neighbors) > 0:
            representation += '['
            for neighbor in self.neighbors:
                representation += str(neighbor) + ','
            representation = representation[:-1] + ']'
        return representation

    def distance(self, other):
        if len(self.neighbors) == 0:
            if self.symbol != other.symbol:
                return 1
            else:
                return 0
        else:
            dist = 0
            number_neighbors = max(len(self.neighbors), len(other.neighbors))
            fraction = 1 / number_neighbors
            for i in range(number_neighbors):
                if i < len(self.neighbors) and i < len(other.neighbors):
                    dist += fraction * self.neighbors[i].distance(other.neighbors[i])
                else:
                    dist += fraction
            dist *= 0.5
            if self.symbol != other.symbol:
                dist += 0.5
            return dist


args = get_arguments()
main(args.input_table)
