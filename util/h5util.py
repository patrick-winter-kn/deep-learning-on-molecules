import h5py
import os


def input_file(path):
    return h5py.File(os.path.expanduser(path), 'r')


def output_file(path):
    return h5py.File(os.path.expanduser(path), 'w')
