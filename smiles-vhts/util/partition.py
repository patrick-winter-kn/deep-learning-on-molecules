import h5py
from progressbar import ProgressBar


def write_partitions(source_file, smiles_matrices_file, partition_names, identifier=None):
    print('Extracting partitions')
    prefix = source_file[:source_file.rfind('.')]
    source_hdf5 = h5py.File(source_file, 'r')
    smiles_matrices_hdf5 = h5py.File(smiles_matrices_file, 'r')
    partition_dataset = 'partition'
    classes_dataset = 'classes'
    if identifier:
        partition_dataset = identifier + '-' + partition_dataset
        classes_dataset = identifier + '-' + classes_dataset
    partition_data = source_hdf5[partition_dataset]
    classes_data = source_hdf5[classes_dataset]
    smiles_matrix_data = smiles_matrices_hdf5['smiles_matrix']
    partition_sizes = analyze_partition_sizes(partition_data)
    target_hdf5s = {}
    target_classes = {}
    target_matrices = {}
    target_i = {}
    print('Found the following partitions:')
    for partition in partition_sizes.keys():
        if partition in partition_names:
            name = partition_names[partition]
        else:
            name = str(partition)
        if identifier:
            name = identifier + '-' + name
        print(name + '(' + str(partition_sizes[partition]) + ')')
        target_hdf5s[partition] = h5py.File(prefix + '-' + name + '.h5', 'w')
        target_matrices[partition] = target_hdf5s[partition]\
            .create_dataset('smiles_matrix', (partition_sizes[partition], smiles_matrix_data.shape[1],
                                              smiles_matrix_data.shape[2]), dtype=smiles_matrix_data.dtype)
        target_classes[partition] = target_hdf5s[partition]\
            .create_dataset('classes', (partition_sizes[partition], classes_data.shape[1]), dtype=classes_data.dtype)
        target_i[partition] = 0
    print('Writing partitions')
    with ProgressBar(max_value=len(partition_data)) as progress:
        for i in range(len(partition_data)):
            partition = partition_data[i][0]
            target_matrices[partition][target_i[partition]] = smiles_matrix_data[i]
            target_classes[partition][target_i[partition]] = classes_data[i]
            target_i[partition] += 1
            progress.update(i + 1)
    for target_hdf5 in target_hdf5s.values():
        target_hdf5.close()
    source_hdf5.close()
    smiles_matrices_hdf5.close()
    print('Finished extracting partitions')


def analyze_partition_sizes(partition_data):
    print('Analyzing partition sizes')
    partition_sizes = {}
    with ProgressBar(max_value=len(partition_data)) as progress:
        for i in range(len(partition_data)):
            partition = partition_data[i][0]
            if partition not in partition_sizes:
                partition_sizes[partition] = 0
            partition_sizes[partition] += 1
            progress.update(i + 1)
    return partition_sizes
