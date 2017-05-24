import h5py


def append_hdf5(file_1, file_2, data_set_name, output_file):
    file_1_hdf5 = h5py.File(file_1, 'r')
    file_2_hdf5 = h5py.File(file_2, 'r')
    output_file_hdf5 = h5py.File(output_file, 'w')
    data_1 = file_1_hdf5[data_set_name]
    data_2 = file_2_hdf5[data_set_name]
    out_shape = list(data_1.shape)
    out_shape[0] += data_2.shape[0]
    out_shape = tuple(out_shape)
    out_data = output_file_hdf5.create_dataset(data_set_name, out_shape, dtype=data_1.dtype)
    out_data[:data_1.shape[0]] = data_1[:]
    out_data[data_1.shape[0]:] = data_2[:]
    file_1_hdf5.close()
    file_2_hdf5.close()
    output_file_hdf5.close()
