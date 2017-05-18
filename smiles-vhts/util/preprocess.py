import numpy
import h5py
from progressbar import ProgressBar


def preprocess(source_file, indices_file, smiles_matrices_file):
    source_hdf5 = h5py.File(source_file, 'r')
    smiles = source_hdf5['smiles']
    characters, max_length = analyze_smiles(smiles)
    indices = write_indices(indices_file, characters)
    write_smiles_matrices(smiles_matrices_file, indices, smiles, max_length)
    source_hdf5.close()


def analyze_smiles(smiles):
    print('Analyzing SMILES')
    max_length = 0
    characters = {' '}
    with ProgressBar(max_value=len(smiles)) as progress:
        i = 0
        for string in smiles:
            string = string.decode('utf-8')
            if max_length < len(string):
                max_length = len(string)
            for character in string:
                characters.add(character)
            i += 1
            progress.update(i)
    return characters, max_length


def write_indices(indices_file, characters):
    print('Writing indices')
    indices_hdf5 = h5py.File(indices_file, 'w')
    index = indices_hdf5.create_dataset('index', (len(characters),), dtype='S1')
    indices = {}
    with ProgressBar(max_value=len(characters)) as progress:
        i = 0
        for character in sorted(characters):
            indices[character] = i
            index[i] = character.encode('utf-8')
            i += 1
            progress.update(i)
    indices_hdf5.close()
    return indices


def write_smiles_matrices(smiles_matrices_file, indices, smiles, max_length):
    print('Writing matrices')
    smiles_matrices_hdf5 = h5py.File(smiles_matrices_file, 'w')
    smiles_matrix = smiles_matrices_hdf5.create_dataset('smiles_matrix', (len(smiles), max_length, len(indices)),
                                                        dtype='I')
    with ProgressBar(max_value=len(smiles)) as progress:
        for i in range(len(smiles)):
            string = pad_string(smiles[i].decode('utf-8'), max_length)
            smiles_matrix[i] = string_to_matrix(string, indices)
            progress.update(i + 1)
    smiles_matrices_hdf5.close()


def pad_string(string, length):
    return string + (' ' * (length - len(string)))


def string_to_matrix(string, indices):
    matrix = numpy.zeros((len(string), len(indices)))
    for i in range(len(string)):
        matrix[i][indices[string[i]]] = 1
    return matrix
