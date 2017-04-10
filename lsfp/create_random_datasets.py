import argparse
import os
import shutil


def get_arguments():
    parser = argparse.ArgumentParser(description='Copies the data sets for use with random position fingerprints')
    parser.add_argument('directory', type=str, help='Directory containing the data sets in subfolders')
    return parser.parse_args()


args = get_arguments()
if not args.directory.endswith('/'):
    args.directory += '/'
dirs = []
names = []
target_dirs = []
for name in os.listdir(args.directory):
    names.append(name)
    dirs.append(args.directory + name + '/')
    target_dirs.append(args.directory + 'r-' + name + '/')
for i in range(len(names)):
    shutil.copytree(dirs[i], target_dirs[i])
    os.rename(target_dirs[i] + names[i] + '.h5', target_dirs[i] + 'r-' + names[i] + '.h5')
