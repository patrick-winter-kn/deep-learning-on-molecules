import h5py
from progressbar import ProgressBar


def write_partitions(source_file, partition_names=None, identifier=None):
    if not partition_names:
        partition_names = {}
    if identifier:
        print('Extracting partitions for data set ' + identifier)
    else:
        print('Extracting partitions')
    prefix = source_file[:source_file.rfind('.')]
    source_hdf5 = h5py.File(source_file, 'r')
    partition_data_set = 'partition'
    if identifier:
        partition_data_set = identifier + '-' + partition_data_set
    partition_data = source_hdf5[partition_data_set]
    partition_sizes = analyze_partition_sizes(partition_data)
    target_hdf5s = {}
    target_refs = {}
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
        target_refs[partition] = target_hdf5s[partition].create_dataset('ref', (partition_sizes[partition],), dtype='I')
        target_i[partition] = 0
    print('Writing partitions')
    with ProgressBar(max_value=len(partition_data)) as progress:
        for i in range(len(partition_data)):
            partition = partition_data[i][0]
            target_refs[partition][target_i[partition]] = i
            target_i[partition] += 1
            progress.update(i + 1)
    for target_hdf5 in target_hdf5s.values():
        target_hdf5.close()
    source_hdf5.close()
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
