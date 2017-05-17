import h5py
from progressbar import ProgressBar
import random


def shuffle(data_file, data_sets=None, seed=42):
    print('Shuffling data in ' + data_file)
    r = random.Random(seed)
    data_hdf5 = h5py.File(data_file, 'r+')
    if not data_sets:
        data_sets = []
        for data_set in data_hdf5.keys():
            data_sets.append(str(data_set))
    n = len(data_hdf5[data_sets[0]])
    with ProgressBar(max_value=n) as progress:
        for i in range(n):
            j = r.randint(0, n-1)
            for data_set in data_sets:
                tmp = data_hdf5[data_set][j]
                data_hdf5[data_set][j] = data_hdf5[data_set][i]
                data_hdf5[data_set][i] = tmp
            progress.update(i+1)
    data_hdf5.close()
