import h5py
from progressbar import ProgressBar


def write_partition(source_file, target_file, smiles_matrices_file, partition):
    print('Extracting partition ' + str(partition))
    source_hdf5 = h5py.File(source_file, 'r')
    target_hdf5 = h5py.File(target_file, 'w')
    smiles_matrices_hdf5 = h5py.File(smiles_matrices_file, 'r')
    partition_data = source_hdf5['partition']
    classes_data = source_hdf5['classes']
    smiles_matrix_data = smiles_matrices_hdf5['smiles_matrix']
    partition_size = analyze_partition_size(partition_data, partition)
    smiles_matrix = target_hdf5.create_dataset('smiles_matrix', (partition_size, smiles_matrix_data.shape[1],
                                                                 smiles_matrix_data.shape[2]),
                                               dtype=smiles_matrix_data.dtype)
    classes = target_hdf5.create_dataset('classes', (partition_size, classes_data.shape[1]), dtype=classes_data.dtype)
    print('Writing partition ' + str(partition) + ' data')
    with ProgressBar(max_value=partition_size) as progress:
        target_i = 0
        for i in range(len(partition_data)):
            if partition == partition_data[i]:
                smiles_matrix[target_i] = smiles_matrix_data[i]
                classes[target_i] = classes_data[i]
                target_i += 1
                progress.update(target_i)
    source_hdf5.close()
    target_hdf5.close()
    smiles_matrices_hdf5.close()


def analyze_partition_size(partition_data, partition):
    print('Analyzing partition size')
    partition_size = 0
    with ProgressBar(max_value=len(partition_data)) as progress:
        for i in range(len(partition_data)):
            if partition == partition_data[i]:
                partition_size += 1
            progress.update(i + 1)
    return partition_size
