import argparse
from util import operations
import os
from datetime import datetime


def get_arguments():
    parser = argparse.ArgumentParser(description='Runs the locality experiment')
    parser.add_argument('directory', type=str, help='Directory containing the data sets in subfolders')
    parser.add_argument('--repeats', type=int, default=5, help='Number of repeated trainings on the same data set')
    return parser.parse_args()


args = get_arguments()
if not args.directory.endswith('/'):
    args.directory += '/'
dirs = []
names = []
for name in os.listdir(args.directory):
    names.append(name)
    dirs.append(args.directory + name + '/')
for i in range(len(dirs)):
    print('='*60)
    print(str(datetime.now())[:-7])
    print('Starting \'' + dirs[i] + '\'')
    print('='*60)
    random = names[i].startswith('r-')
    operations.prepare_data(dirs[i], names[i], 2, random)
    for j in range(args.repeats):
        operations.train_model(dirs[i], names[i], j, 5, 200)
        operations.predict(dirs[i], names[i], j, 200)
    # Mark that we are done with this data set
    open(dirs[i] + 'done', 'a').close()
    print('='*60)
    print(str(datetime.now())[:-7])
    print('Finished \'' + dirs[i] + '\'')
    print('='*60)
