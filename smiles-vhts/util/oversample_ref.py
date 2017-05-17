import os
import h5py
from progressbar import ProgressBar
import math


def oversample(refs_file, data_file, identifier=None):
    print('Oversampling data in ' + refs_file)
    # This is currently only implemented for 2 classes
    prefix = refs_file[:refs_file.rfind('.')]
    refs_hdf5 = h5py.File(refs_file, 'r')
    ref = refs_hdf5['ref']
    source_hdf5 = h5py.File(data_file, 'r')
    target_file = prefix + '-oversampled.h5'
    target_hdf5 = h5py.File(target_file, 'w')
    classes_data_set = 'classes'
    if identifier:
        classes_data_set = identifier + '-' + classes_data_set
    classes = source_hdf5[classes_data_set]
    class_zero_count = 0
    class_one_count = 0
    for i in range(ref.shape[0]):
        value = classes[ref[i]]
        if value[0] >= value[1]:
            class_zero_count += 1
        else:
            class_one_count += 1
    difference = abs(class_zero_count - class_one_count)
    oversampled_ref = target_hdf5.create_dataset('ref', (ref.shape[0] + difference,), dtype='I')
    left_difference = difference
    with ProgressBar(max_value=oversampled_ref.shape[0]) as progress:
        if class_zero_count < class_one_count:
            copies_per_instance = math.ceil(class_one_count / class_zero_count)
        else:
            copies_per_instance = math.ceil(class_zero_count / class_one_count)
        target_i = 0
        for i in range(len(ref)):
            value = classes[ref[i]]
            minority = (class_zero_count < class_one_count and value[0] >= value[1]) or \
                       (class_one_count < class_zero_count and value[1] > value[0])
            copies = 1
            if left_difference > 0 and minority:
                copies = min(left_difference + 1, copies_per_instance)
                left_difference -= copies - 1
            for j in range(copies):
                oversampled_ref[target_i] = ref[i]
                target_i += 1
                progress.update(target_i)
    source_hdf5.close()
    target_hdf5.close()
    refs_hdf5.close()
    os.remove(refs_file)
    os.rename(target_file, refs_file)
