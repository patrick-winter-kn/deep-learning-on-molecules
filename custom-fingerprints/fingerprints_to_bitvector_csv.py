import argparse
import h5py
from progressbar import ProgressBar


def get_arguments():
    parser = argparse.ArgumentParser(description='Convert fingerprint HDF5 file to bitvector csv file')
    parser.add_argument('hdf5_file_path', type=str, help='Path to the fingerprint HDF5 file')
    return parser.parse_args()


args = get_arguments()
in_path = args.hdf5_file_path
out_path = in_path[:in_path.rfind('.')] + '.csv'
in_file = h5py.File(in_path, 'r')
in_data = in_file['fingerprint']
with open(out_path, 'w') as out_file:
    with ProgressBar(max_value=in_data.shape[0]) as progress:
        i = 0
        for fingerprint in in_data:
            string = ''
            for value in fingerprint:
                if value > 0:
                    string += '1'
                else:
                    string += '0'
            out_file.write(string + '\n')
            i += 1
            progress.update(i)
in_file.close()
