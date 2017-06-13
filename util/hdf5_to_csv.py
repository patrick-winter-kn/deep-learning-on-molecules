import h5py
import numpy
import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description='Export a HDF5 data set to CSV')
    parser.add_argument('input_file', type=str, help='Input file')
    parser.add_argument('data_set_name', type=str, help='Name of the data set')
    parser.add_argument('output_file', type=str, help='Output file')
    return parser.parse_args()


args = get_arguments()
h5_file = h5py.File(args.input_file, 'r')
numpy.savetxt(args.output_file, h5_file[args.data_set_name], '%g', ',')
h5_file.close()
