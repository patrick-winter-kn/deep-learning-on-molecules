import argparse
import h5py


def get_arguments():
    parser = argparse.ArgumentParser(description='Prepare training data')
    parser.add_argument('fingerprints', type=str, help='Path to HDF5 file containing the fingerprints')
    parser.add_argument('classes', type=str, help='Path to HDF5 file containing the classes.')
    return parser.parse_args()


args = get_arguments()
prefix = args.fingerprints[:args.fingerprints.rfind('.')]
fingerprints_file = h5py.File(args.fingerprints, 'r')
classes_file = h5py.File(args.classes, 'r')
train_file = h5py.File(prefix + '-train.h5', 'w')
indices = classes_file['index']
fingerprints = fingerprints_file['fingerprint']
train_file.create_dataset('class', data=classes_file['class'])
fingerprint_data = train_file.create_dataset('fingerprint', (indices.shape[0], fingerprints.shape[1]), dtype='I')
for i in range(indices.shape[0]):
    fingerprint_data[i] = fingerprints[indices[i][0]]
fingerprints_file.close()
classes_file.close()
train_file.close()