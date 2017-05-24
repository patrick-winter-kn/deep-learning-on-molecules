from functools import total_ordering
import h5py
from progressbar import ProgressBar
import numpy
import random
from rdkit import Chem
#from tsp_solver.greedy import  solve_tsp
from sklearn.manifold import MDS


def fingerprint_with_new_index(smiles_file_path, random_order):
    prefix = smiles_file_path[:smiles_file_path.rfind('.')]
    smiles_file = h5py.File(smiles_file_path, 'r')
    neighborhoods_file = h5py.File(prefix + '-neighborhoods.h5', 'w')
    index_file = h5py.File(prefix + '-index.h5', 'w')
    fingerprint_file = h5py.File(prefix + '-fingerprint.h5', 'w')
    neighborhoods_set = extract_neighborhoods(smiles_file, neighborhoods_file)
    write_index_positions(neighborhoods_set, index_file, random_order)
    generate_fingerprints(neighborhoods_file, index_file, fingerprint_file)
    smiles_file.close()
    neighborhoods_file.close()
    index_file.close()
    fingerprint_file.close()


def fingerprint_with_existing_index(smiles_file_path, index_file_path):
    prefix = smiles_file_path[:smiles_file_path.rfind('.')]
    smiles_file = h5py.File(smiles_file_path, 'r')
    index_file = h5py.File(index_file_path, 'r')
    neighborhoods_file = h5py.File(prefix + '-neighborhoods.h5', 'w')
    fingerprint_file = h5py.File(prefix + '-fingerprint.h5', 'w')
    extract_neighborhoods(smiles_file, neighborhoods_file)
    generate_fingerprints(neighborhoods_file, index_file, fingerprint_file)
    smiles_file.close()
    index_file.close()
    neighborhoods_file.close()
    fingerprint_file.close()


def extract_neighborhoods(smiles_file, neighborhoods_file, radius):
    print('Extracting neighborhoods')
    smiles_hdf5 = h5py.File(smiles_file, 'r')
    neighborhoods_hdf5 = h5py.File(neighborhoods_file, 'w')
    smiles = smiles_hdf5['smiles']
    neighborhoods_out = neighborhoods_hdf5.create_dataset('neighborhoods', (len(smiles),), 'S2000')
    with ProgressBar(max_value=len(smiles)) as progress:
        for i in range(len(smiles)):
            neighborhoods = neighborhoods_from_smiles(smiles[i], radius)
            neighborhoods_out[i] = neighborhoods_to_string(neighborhoods).encode('utf-8')
            progress.update(i)
    smiles_hdf5.close()
    neighborhoods_hdf5.close()



def write_index_positions(neighborhoods_file, index_file, random_order):
    neighborhoods_hdf5 = h5py.File(neighborhoods_file, 'r')
    index_hdf5 = h5py.File(index_file, 'w')
    neighborhoods_set = set()
    print('Creating set of neighborhoods')
    i = 0
    with ProgressBar(max_value=len(neighborhoods_hdf5['neighborhoods'])) as progress:
        for neighborhoods_string in neighborhoods_hdf5['neighborhoods']:
            neighborhoods = neighborhoods_from_string(neighborhoods_string.decode('utf-8'))
            for n in neighborhoods:
                neighborhoods_set.add(n)
            i += 1
            progress.update(i)
    index_out = index_hdf5.create_dataset('index', (len(neighborhoods_set),), 'S2000')
    i = 0
    if random_order:
        neighborhoods_sorted = randomized_neighborhoods(neighborhoods_set)
    else:
        neighborhoods_sorted = sorted_neighborhoods(neighborhoods_set)
    print('Writing index positions')
    with ProgressBar(max_value=len(neighborhoods_set)) as progress:
        for neighborhood in neighborhoods_sorted:
            index_out[i] = str(neighborhood).encode('utf-8')
            i += 1
            progress.update(i)
    neighborhoods_hdf5.close()
    index_hdf5.close()


def generate_fingerprints(neighborhoods_file, index_file, fingerprint_file):
    neighborhoods_hdf5 = h5py.File(neighborhoods_file, 'r')
    index_hdf5 = h5py.File(index_file, 'r')
    fingerprint_hdf5 = h5py.File(fingerprint_file, 'w')
    index_dict = dict_from_file(index_hdf5)
    print('Generating fingerprints')
    neighborhoods_out = neighborhoods_hdf5['neighborhoods']
    index_out = index_hdf5['index']
    fingerprint_out = fingerprint_hdf5.create_dataset('fingerprint', (neighborhoods_out.shape[0], index_out.shape[0]),
                                                      'I')
    with ProgressBar(max_value=neighborhoods_out.shape[0]) as progress:
        for i in range(neighborhoods_out.shape[0]):
            array = numpy.zeros(index_out.shape[0], dtype=numpy.uint32)
            neighborhoods = neighborhoods_from_string(neighborhoods_out[i].decode('utf-8'))
            for neighborhood in neighborhoods:
                if neighborhood in index_dict:
                    index = index_dict[neighborhood]
                    array[index] = neighborhoods[neighborhood]
            fingerprint_out[i] = array
            progress.update(i+1)
    neighborhoods_hdf5.close()
    index_hdf5.close()
    fingerprint_hdf5.close()


def dict_from_file(file):
    print('Creating lookup table')
    index = file['index']
    index_dict = {}
    with ProgressBar(max_value=index.shape[0]) as progress:
        for i in range(index.shape[0]):
            index_dict[Neighborhood.from_string(index[i].decode('utf-8'))] = i
            progress.update(i+1)
    return index_dict


def neighborhoods_from_smiles(smiles, radius):
    mol = Chem.MolFromSmiles(smiles)
    neighborhoods = {}
    for atom in mol.GetAtoms():
        atom_neighborhood = Neighborhood.from_atom(atom, radius, set())
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


def sorted_neighborhoods(neighborhoods_set):
    return mds_sorted_neighborhoods(neighborhoods_set)


def mds_sorted_neighborhoods(neighborhoods_set):
    neighborhoods = list(neighborhoods_set)
    matrix = numpy.zeros(shape=(len(neighborhoods), len(neighborhoods)))
    print('Calculating distance matrix')
    counter = 0
    with ProgressBar(max_value=int(len(neighborhoods)*len(neighborhoods)-len(neighborhoods))/2) as progress:
        for i in range(len(neighborhoods)):
            for j in range(i+1, len(neighborhoods)):
                matrix[i][j] = neighborhoods[i].distance(neighborhoods[j])
                matrix[j][i] = matrix[i][j]
                counter += 1
                progress.update(counter)
    print('Sorting based on distances')
    mds = MDS(n_components=1, dissimilarity='precomputed')
    results = mds.fit_transform(matrix)
    sorted = []
    for index in results.argsort(axis=None):
        sorted.append(neighborhoods[index])
    return sorted


# def tsp_sorted_neighborhoods(neighborhoods_set):
#     neighborhoods = list(neighborhoods_set)
#     matrix = numpy.zeros(shape=(len(neighborhoods), len(neighborhoods)))
#     print('Calculating distance matrix')
#     counter = 0
#     with ProgressBar(max_value=(len(neighborhoods)*len(neighborhoods)-len(neighborhoods))/2) as progress:
#         for i in range(len(neighborhoods)):
#             for j in range(i+1, len(neighborhoods)):
#                 matrix[i][j] = neighborhoods[i].distance(neighborhoods[j])
#                 counter += 1
#                 progress.update(counter)
#     print('Sorting based on distances')
#     path = solve_tsp(matrix)
#     sorted = []
#     for i in path:
#         sorted.append(neighborhoods[i])
#     return sorted


def greedy_sorted_neighborhoods(neighborhoods_set):
    print('Sorting neighborhoods greedily')
    neighborhoods = list(neighborhoods_set)
    sorted = []
    if neighborhoods:
        with ProgressBar(max_value=(len(neighborhoods)*len(neighborhoods)-len(neighborhoods))/2) as progress:
            sorted.append(neighborhoods[0])
            del neighborhoods[0]
            counter = 0
            while neighborhoods:
                tail = sorted[len(sorted)-1]
                min_distance = tail.distance(neighborhoods[0])
                min_distance_index = 0
                counter += 1
                progress.update(counter)
                for i in range(1, len(neighborhoods)):
                    distance = tail.distance(neighborhoods[i])
                    if distance < min_distance:
                        min_distance = distance
                        min_distance_index = i
                    counter += 1
                    progress.update(counter)
                sorted.append(neighborhoods[min_distance_index])
                del neighborhoods[min_distance_index]
    return sorted


def randomized_neighborhoods(neighborhoods_set):
    neighborhoods = list(neighborhoods_set)
    random.shuffle(neighborhoods)
    return neighborhoods


def evaluate_index_locality(data, index, out, radius):
    data_file = h5py.File(data, 'r')
    index_file = h5py.File(index, 'r')
    out_file = h5py.File(out, 'w')
    smiles = data_file['smiles']
    index = index_file['index']
    results = {}
    index_lookup = {}
    for i in range(index.shape[0]):
        index_lookup[index[i].decode('utf-8')] = i
    print('Calculating position differences')
    with ProgressBar(max_value=smiles.shape[0]) as progress:
        for i in range(smiles.shape[0]):
            seen = set()
            mol = Chem.MolFromSmiles(smiles[i].decode('utf-8'))
            for atom in mol.GetAtoms():
                atom_neighborhood = str(Neighborhood.from_atom(atom, radius, set()))
                for neighbor_atom in atom.GetNeighbors():
                    atom_pair = to_tuple(atom, neighbor_atom)
                    # Skip ones that we have already seen from the other side
                    if atom_pair not in seen:
                        seen.add(atom_pair)
                        neighbor_neighborhood = str(Neighborhood.from_atom(neighbor_atom, radius, set()))
                        atom_position = index_lookup[atom_neighborhood]
                        neighbor_position = index_lookup[neighbor_neighborhood]
                        difference = abs(atom_position - neighbor_position)
                        if difference in results:
                            results[difference] += 1
                        else:
                            results[difference] = 1
            progress.update(i+1)
    out = out_file.create_dataset('position_difference', shape=(sum(results.values()),1))
    i = 0
    print('Writing results')
    with ProgressBar(max_value=out.shape[0]) as progress:
        for difference in results:
            for j in range(results[difference]):
                out[i] = difference
                i += 1
                progress.update(i)
    data_file.close()
    index_file.close()
    out_file.close()


def to_tuple(atom_1, atom_2):
    if atom_1.GetIdx() <= atom_2.GetIdx():
        return (atom_1.GetIdx(), atom_2.GetIdx())
    else:
        return (atom_2.GetIdx(), atom_1.GetIdx())


@total_ordering
class Neighborhood:

    def __init__(self, symbol, neighbors):
        self.symbol = symbol
        self.neighbors = sorted(neighbors)
        self.atom_pairs = {}
        self.add_atom_pairs(self.atom_pairs)

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
        if '(' not in string:
            return cls(string, [])
        else:
            symbol = string[:string.find('(')]
            string = string[string.find('('):]
            neighbors = []
            for split in find_splits(string):
                neighbors.append(Neighborhood.from_string(split))
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
            representation += '('
            for neighbor in self.neighbors:
                representation += str(neighbor) + ')('
            representation = representation[:-1]
        return representation

    def distance(self, other):
        max_number_pairs = max(sum(self.atom_pairs.values()), sum(other.atom_pairs.values()))
        if max_number_pairs < 1:
            if self.symbol == other.symbol:
                return 0
            else:
                return 1
        pair_intersection = self.atom_pairs.keys() & other.atom_pairs.keys()
        distance = max_number_pairs
        for pair in pair_intersection:
            distance -= min(self.atom_pairs[pair], other.atom_pairs[pair])
        return distance / max_number_pairs

    def add_atom_pairs(self, atom_pairs):
        for i in range(len(self.neighbors)):
            pair = sorted([self.symbol, self.neighbors[i].symbol])
            pair_string = pair[0] + '-' + pair[1]
            if pair_string in atom_pairs:
                atom_pairs[pair_string] += 1
            else:
                atom_pairs[pair_string] = 1
            self.neighbors[i].add_atom_pairs(atom_pairs)

def find_splits(string):
    splits = []
    start = 1
    depth = 0
    for i in range(len(string)):
        character = string[i]
        if character == '(':
            depth += 1
        elif character == ')':
            depth -= 1
        if depth == 0:
            splits.append(string[start : i])
            start = i + 2
    return splits
