from h5py._hl import dataset


class ReferenceDataSet(dataset.Dataset):

    def __init__(self, reference_data_set, data_data_set):
        self.reference = reference_data_set
        self.data = data_data_set
        super().__init__(data_data_set._id)

    def __len__(self):
        return len(self.reference)

    def __getitem__(self, item):
        if isinstance(item, slice):
            start = item.start
            if not start:
                start = 0
            stop = item.stop
            if not stop:
                stop = len(self.reference)
            item = slice(start, stop, None)
            maximum = max(item.start, item.stop)
            return [self.data[self.reference[i]] for i in range(*item.indices(maximum+1))]
        elif isinstance(item, list):
            return [self.data[self.reference[i]] for i in item]
        else:
            return self.data[self.reference[item]]

    @property
    def shape(self):
        shape_list = list(self.data.shape)
        shape_list[0] = self.reference.shape[0]
        return tuple(shape_list)
