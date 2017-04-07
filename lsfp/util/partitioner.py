import h5py

def create_partition(source_file, fingerprints_file, partition_file, partition_value):
    print('Extracting partition ' + str(partition_value))
    source_hdf5 = h5py.File(source_file, 'r')
    fingerprints_hdf5 = h5py.File(fingerprints_file, 'r')
    partition_hdf5 = h5py.File(partition_file, 'w')
    partitions = source_hdf5['partition']
    classes = source_hdf5['classes']
    fingerprints = fingerprints_hdf5['fingerprint']
    partition_size = 0
    for i in range(partitions.shape[0]):
        if partitions[i] == partition_value:
            partition_size += 1
    partition_fingerprints = partition_hdf5.create_dataset('fingerprint', (partition_size, fingerprints.shape[1]))
    partition_classes = partition_hdf5.create_dataset('classes', (partition_size, classes.shape[1]))
    partition_i = 0
    for i in range(partitions.shape[0]):
        if partitions[i] == partition_value:
            partition_fingerprints[partition_i] = fingerprints[i]
            partition_classes[partition_i] = classes[i]
            partition_i += 1
    source_hdf5.close()
    fingerprints_hdf5.close()
    partition_hdf5.close()
