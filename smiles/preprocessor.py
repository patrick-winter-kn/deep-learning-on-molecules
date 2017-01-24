import numpy
from tqdm import tnrange
from itertools import chain


_all_chars_ = ' []()*-=#$:/\\.%@+0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_full_charset_ = sorted(list(chain.from_iterable(_all_chars_)))


def preprocess(in_file, out_file, smiles_column_name='structure', max_length=180, charset=_full_charset_):
    # get smiles
    smiles = convert_bytes_to_strings(in_file['table']['table'][smiles_column_name])
    # if max_length is none, find the biggest string in the data
    if max_length is None:
        max_length = find_max_length(smiles)
    # pad and cut smiles to the correct size
    pad_strings(smiles, max_length)
    # if charset is None extract the charset from the data
    if charset is None:
        charset = extract_charset(smiles)
    # write charset
    out_file.create_dataset('charset', data=numpy.array(charset).astype(bytes))
    # encode smiles
    write_encoded_smiles(out_file, 'data', smiles, max_length, charset)


def find_max_length(strings):
    max_length = 0
    for i in range(0, len(strings)):
        max_length = max(max_length, strings[i])
    return max_length


def pad_strings(strings, length):
    for i in range(0, len(strings)):
        strings[i] = (strings[i] + spaces(length-len(strings[i])))[:length]


def spaces(n):
    string = ''
    for i in range(0, n):
        string += ' '
    return string


def extract_charset(smiles_strings):
    charset = set()
    for smiles in smiles_strings:
        smiles_string = smiles
        for character in smiles_string:
            charset.add(character)
    return sorted(charset)


def write_encoded_smiles(file, name, smiles, smiles_length, charset):
    data_set = file.create_dataset(name, (len(smiles), smiles_length, len(charset)))
    for i in tnrange(len(smiles)):
        for j in range(len(smiles[i])):
            data_set[i, j] = encode_char(smiles[i][j], charset)


def encode_char(character, charset):
    bit_array = numpy.ndarray(len(charset))
    for i in range(len(charset)):
        if character == charset[i]:
            bit_array[i] = 1
        else:
            bit_array[i] = 0
    return bit_array


def convert_bytes_to_strings(byte_strings):
    strings = numpy.ndarray(shape=(len(byte_strings)), dtype=object)
    for i in range(len(strings)):
        strings[i] = byte_strings[i].decode('utf-8')
    return strings
