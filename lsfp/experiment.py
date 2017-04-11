import argparse
from util import operations
import os
from datetime import datetime


def get_arguments():
    parser = argparse.ArgumentParser(description='Runs the locality experiment')
    parser.add_argument('directory', type=str, help='Directory containing the data sets in subfolders')
    parser.add_argument('--repeats', type=int, default=5, help='Number of repeated trainings on the same data set')
    return parser.parse_args()


def delete_everything_but(directory, exceptions):
    for file in os.listdir(directory):
        if file not in exceptions:
            os.remove(directory + file)


def find_first_free_id(directory, name):
    id = 0
    while os.path.isfile(directory + name + '-' + str(id) + '-' + operations._model_suffix_ + '.h5'):
        id += 1
    return id


args = get_arguments()
if not args.directory.endswith('/'):
    args.directory += '/'
dirs = []
names = []
for name in sorted(os.listdir(args.directory)):
    names.append(name)
    dirs.append(args.directory + name + '/')
for i in range(len(dirs)):
    print('='*60)
    print(str(datetime.now())[:-7])
    print('Starting \'' + dirs[i] + '\'')
    print('='*60)
    random = names[i].startswith('r-')
    if not os.path.isfile(dirs[i] + 'preprocessing-done'):
        # Cleanup files from a potentially failed previous preprocessing
        delete_everything_but(dirs[i], {names[i]+'.h5'})
        operations.prepare_data(dirs[i], names[i], 2, random)
        open(dirs[i] + 'preprocessing-done', 'a').close()
    id_offset = find_first_free_id(dirs[i], names[i])
    for id in range(id_offset, args.repeats + id_offset):
        operations.train_model(dirs[i], names[i], id, 5, 200)
        operations.predict(dirs[i], names[i], id, 200)
    print('='*60)
    print(str(datetime.now())[:-7])
    print('Finished \'' + dirs[i] + '\'')
    print('='*60)
