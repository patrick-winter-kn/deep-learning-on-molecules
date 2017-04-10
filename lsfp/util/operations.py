from os import path
from util import partitioner
from util import preprocessor
try:
    from util import neighborhoods
except ImportError:
    pass
try:
    from util import learner
except ImportError:
    pass


_neighborhoods_suffix_ = 'neighborhoods'
_indices_suffix_ = 'indices'
_fingerprints_suffix_ = 'fingerprints'
_train_suffix_ = 'train'
_test_suffix_ = 'test'
_model_suffix_ = 'model'
_predictions_suffix_= 'predictions'


def prepare_data(directory, name, radius, random):
    create_neighborhoods(directory, name, radius)
    create_indices(directory, name, random)
    create_fingerprints(directory, name)
    create_train(directory, name)
    create_test(directory, name)
    oversample(directory, name)


def train_model(directory, name, id, epochs, batch_size):
    learner.train_model(file_path(directory, name, _train_suffix_), file_path(directory, name, _model_suffix_, id),
                        epochs, batch_size)


def predict(directory, name, id, batch_size):
    if exists(directory, name, _predictions_suffix_, id):
        return
    learner.predict(file_path(directory, name, _test_suffix_), file_path(directory, name, _model_suffix_, id),
                    file_path(directory, name, _predictions_suffix_, id), batch_size)


def create_neighborhoods(directory, name, radius):
    if exists(directory, name, _neighborhoods_suffix_):
        return
    neighborhoods.extract_neighborhoods(file_path(directory, name), file_path(directory, name, _neighborhoods_suffix_),
                                        radius)


def create_indices(directory, name, random):
    if exists(directory, name, _indices_suffix_):
        return
    neighborhoods.write_index_positions(file_path(directory, name, _neighborhoods_suffix_),
                                        file_path(directory, name,_indices_suffix_), random)


def create_fingerprints(directory, name):
    if exists(directory, name, _fingerprints_suffix_):
        return
    neighborhoods.generate_fingerprints(file_path(directory, name, _neighborhoods_suffix_),
                                        file_path(directory, name,_indices_suffix_),
                                        file_path(directory, name, _fingerprints_suffix_))


def create_train(directory, name):
    if exists(directory, name, _train_suffix_):
        return
    partitioner.create_partition(file_path(directory, name),
                                 file_path(directory, name,_fingerprints_suffix_),
                                 file_path(directory, name, _train_suffix_),
                                 1)


def create_test(directory, name):
    if exists(directory, name, _test_suffix_):
        return
    partitioner.create_partition(file_path(directory, name),
                                 file_path(directory, name,_fingerprints_suffix_),
                                 file_path(directory, name, _test_suffix_),
                                 2)


def oversample(directory, name):
    preprocessor.oversample(file_path(directory, name, _train_suffix_))


def file_path(directory, name, suffix=None, id=None):
    if directory.endswith('/'):
        directory = directory[:-1]
    if id is not None:
        name += '-' + str(id)
    if suffix is not None:
        name += '-' + suffix
    return directory + '/' + name + '.h5'


def exists(directory, name, suffix=None, id=None):
    return path.isfile(file_path(directory, name, suffix, id))
