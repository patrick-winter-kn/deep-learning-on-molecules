import numpy
from progressbar import ProgressBar
from itertools import chain
import threading
import multiprocessing
import math


class EncoderThread (threading.Thread):

    thread_lock = threading.Lock()
    update_after_n = 100

    def __init__(self, data, charset, smiles_length, progress):
        threading.Thread.__init__(self)
        self.data = data
        self.charset = charset
        self.smiles_length = smiles_length
        self.progress = progress
        self.encoded_data = None

    def run(self):
        smiles = self.data
        charset = self.charset
        smiles_length = self.smiles_length
        data_set = numpy.ndarray(shape=(len(smiles), smiles_length, len(charset)))
        for i in range(len(smiles)):
            for j in range(len(smiles[i])):
                data_set[i, j] = encode_char(smiles[i][j], charset)
            if i % EncoderThread.update_after_n is EncoderThread.update_after_n - 1:
                EncoderThread.thread_lock.acquire()
                self.progress.increment(EncoderThread.update_after_n)
                EncoderThread.thread_lock.release()
        self.encoded_data = data_set

    def get_encoded_data(self):
        return self.encoded_data


class Progress:

    def __init__(self, n):
        self.i = 0
        self.bar = ProgressBar(max_value=n)

    def increment(self, n):
        self.i += n
        self.bar.update(self.i)

    def finish(self):
        self.bar.finish()


_all_chars_ = ' []()*-=#$:/\\.%@+0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_full_charset_ = sorted(list(chain.from_iterable(_all_chars_)))


def preprocess(in_file, out_file, smiles_column_name='structure', max_length=120, charset=_full_charset_):
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
    progress = Progress(len(smiles))
    threads = []
    number_chunks = multiprocessing.cpu_count()
    rows_per_chunk = math.ceil(len(smiles) / number_chunks)
    for i in range(number_chunks):
        start = rows_per_chunk * i
        end = min(start + rows_per_chunk, len(smiles))
        thread = EncoderThread(smiles[start:end], charset, smiles_length, progress)
        thread.start()
        threads.append(thread)
    data_set = numpy.ndarray(shape=(0, smiles_length, len(charset)))
    for thread in threads:
        thread.join()
        data_set = numpy.concatenate((data_set, thread.get_encoded_data()), axis=0)
    file.create_dataset(name, data=data_set)
    progress.finish()


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
