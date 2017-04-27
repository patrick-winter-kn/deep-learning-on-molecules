import os
import h5py
from progressbar import ProgressBar
import math


def oversample(data_file):
    print('Oversampling data in ' + data_file)
    # This is currently only implemented for 2 classes
    prefix = data_file[:data_file.rfind('.')]
    source_hdf5 = h5py.File(data_file, 'r')
    target_file = prefix + '-oversampled.h5'
    target_hdf5 = h5py.File(target_file, 'w')
    classes = source_hdf5['classes']
    smiles_matrix = source_hdf5['smiles_matrix']
    class_zero_count = 0
    class_one_count = 0
    for value in classes:
        if value[0] >= value[1]:
            class_zero_count += 1
        else:
            class_one_count += 1
    difference = abs(class_zero_count - class_one_count)
    oversampled_classes = target_hdf5.create_dataset('classes', (classes.shape[0] + difference, classes.shape[1]))
    oversampled_smiles_matrix = target_hdf5.create_dataset('smiles_matrix', (smiles_matrix.shape[0] + difference,
                                                                    smiles_matrix.shape[1], smiles_matrix.shape[2]))
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
                oversampled_smiles_matrix[target_i] = smiles_matrix[i]
                target_i += 1
                progress.update(target_i)
    source_hdf5.close()
    target_hdf5.close()
    os.remove(data_file)
    os.rename(target_file, data_file)
