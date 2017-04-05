import argparse
import h5py
import math
from progressbar import ProgressBar


def get_arguments():
    parser = argparse.ArgumentParser(description='Oversample underrepresented classes.')
    parser.add_argument('classes', type=str, help='Path to HDF5 file containing the classes.')
    return parser.parse_args()


# This is currently only implemented for 2 classes
args = get_arguments()
prefix = args.classes[:args.classes.rfind('.')]
source_file = h5py.File(args.classes, 'r')
target_file = h5py.File(prefix + '-oversampled.h5', 'w')
classes = source_file['class']
indices = source_file['index']
class_zero_count = 0
class_one_count = 0
for value in classes:
    if value[0] >= value[1]:
        class_zero_count += 1
    else:
        class_one_count += 1
difference = abs(class_zero_count - class_one_count)
oversampled_classes = target_file.create_dataset('class', (classes.shape[0] + difference, classes.shape[1]))
oversampled_indices = target_file.create_dataset('index', (indices.shape[0] + difference, indices.shape[1]))
left_difference = difference
with ProgressBar(max_value=oversampled_classes.shape[0]) as progress:
    if class_zero_count < class_one_count:
        copies_per_instance = math.ceil(class_one_count / class_zero_count)
    else:
        copies_per_instance = math.ceil(class_zero_count / class_one_count)
    target_i = 0
    for i in range(len(classes)):
        minority = (class_zero_count < class_one_count and classes[i][0] >= classes[i][1]) or\
                   (class_one_count < class_zero_count and classes[i][1] > classes[i][0])
        copies = 1
        if left_difference > 0 and minority:
            copies = min(left_difference + 1, copies_per_instance)
            left_difference -= copies - 1
        for j in range(copies):
            oversampled_classes[target_i] = classes[i]
            oversampled_indices[target_i] = indices[i]
            target_i += 1
            progress.update(target_i)
source_file.close()
target_file.close()
