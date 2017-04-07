import os
import h5py
from progressbar import ProgressBar
import math


def oversample(train_file):
    print('Oversampling training data')
    # This is currently only implemented for 2 classes
    prefix = train_file[:train_file.rfind('.')]
    source_hdf5 = h5py.File(train_file, 'r')
    target_file = prefix + '-oversampled.h5'
    target_hdf5 = h5py.File(target_file, 'w')
    classes = source_hdf5['classes']
    fingerprints = source_hdf5['fingerprint']
    class_zero_count = 0
    class_one_count = 0
    for value in classes:
        if value[0] >= value[1]:
            class_zero_count += 1
        else:
            class_one_count += 1
    difference = abs(class_zero_count - class_one_count)
    oversampled_classes = target_hdf5.create_dataset('classes', (classes.shape[0] + difference, classes.shape[1]))
    oversampled_fingerprints = target_hdf5.create_dataset('fingerprint', (fingerprints.shape[0] + difference,
                                                                    fingerprints.shape[1]))
    left_difference = difference
    with ProgressBar(max_value=oversampled_classes.shape[0]) as progress:
        if class_zero_count < class_one_count:
            copies_per_instance = math.ceil(class_one_count / class_zero_count)
        else:
            copies_per_instance = math.ceil(class_zero_count / class_one_count)
        target_i = 0
        for i in range(len(classes)):
            minority = (class_zero_count < class_one_count and classes[i][0] >= classes[i][1]) or \
                       (class_one_count < class_zero_count and classes[i][1] > classes[i][0])
            copies = 1
            if left_difference > 0 and minority:
                copies = min(left_difference + 1, copies_per_instance)
                left_difference -= copies - 1
            for j in range(copies):
                oversampled_classes[target_i] = classes[i]
                oversampled_fingerprints[target_i] = fingerprints[i]
                target_i += 1
                progress.update(target_i)
    source_hdf5.close()
    target_hdf5.close()
    os.remove(train_file)
    os.rename(target_file, train_file)
