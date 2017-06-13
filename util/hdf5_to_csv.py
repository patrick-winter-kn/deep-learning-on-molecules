import h5py
import numpy
import argparse
import math
from progressbar import ProgressBar
import gzip


def get_arguments():
    parser = argparse.ArgumentParser(description='Export a HDF5 data set to CSV')
    parser.add_argument('input_file', type=str, help='Input file')
    parser.add_argument('data_set_name', type=str, help='Name of the data set')
    parser.add_argument('output_file', type=str, help='Output file')
    parser.add_argument('--batch_size', type=int, default=1000, help='Size of a batch (default: 1000)')
    return parser.parse_args()


args = get_arguments()
h5_file = h5py.File(args.input_file, 'r')
data_set = h5_file[args.data_set_name]
if args.output_file.endswith('.gz'):
    out_file = gzip.open(args.output_file, 'wb')
else:
    out_file = open(args.output_file, 'wb')
iterations = int(math.ceil(len(data_set) / args.batch_size))
with ProgressBar(max_value=len(data_set)) as progress:
    for i in range(iterations):
        start = i * args.batch_size
        end = min(len(data_set), start + args.batch_size)
        numpy.savetxt(out_file, data_set[start:end], '%g', ',')
        progress.update(end)
h5_file.close()
out_file.close()
