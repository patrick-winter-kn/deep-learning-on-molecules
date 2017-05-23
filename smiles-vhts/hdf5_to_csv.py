import numpy
import h5py
import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description='Write a data set contained in the HDF5 file into a CSV file')
    parser.add_argument('input_hdf5', type=str, help='Input file in HDF5 format')
    parser.add_argument('data_set', type=str, help='Name of the data set to export')
    parser.add_argument('output_csv', type=str, help='Output file in CSV format. Will be automatically zipped if the '
                                                     'name ends with .gz')
    return parser.parse_args()

args = get_arguments()
data_h5 = h5py.File(args.input_hdf5, 'r')
numpy.savetxt(args.output_csv, data_h5[args.data_set], '%g', ',')
data_h5.close()
