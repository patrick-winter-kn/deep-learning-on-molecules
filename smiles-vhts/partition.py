import argparse
import h5py
from progressbar import ProgressBar
import random


def get_arguments():
    parser = argparse.ArgumentParser(description='Partition data')
    parser.add_argument('file', type=str, help='The file containing the classes')
    parser.add_argument('split', type=float, help='Percentage of data in the training partition')
    parser.add_argument('--dataset_prefix', type=str, default='', help='Prefix of the dataset holding the activity data (default: None)')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    return parser.parse_args()


args = get_arguments()
random.seed(args.seed)
data_h5 = h5py.File(args.file, 'r+')
classes = data_h5[args.dataset_prefix + 'classes']
active_indices = []
inactive_indices = []
print('Retrieving active and inactive data')
with ProgressBar(max_value=len(classes)) as progress:
    for i in range(len(classes)):
        if classes[i,0] > 0.0:
            active_indices.append(i)
        else:
            inactive_indices.append(i)
        progress.update(i+1)
print('Found ' + str(len(active_indices)) + ' active indices and ' + str(len(inactive_indices)) + ' inactive data points')
number_training = round(len(classes) * args.split * 0.01)
number_training_active = round(number_training * (len(active_indices) / len(classes)))
number_training_inactive = number_training - number_training_active
print('Picking data points for training')
with ProgressBar(max_value=number_training) as progress:
    for i in range(number_training_active):
        del active_indices[random.randint(0, len(active_indices) - 1)]
        progress.update(i+1)
    for i in range(number_training_inactive):
        del inactive_indices[random.randint(0, len(inactive_indices) - 1)]
        progress.update(i + number_training_active + 1)
partition = data_h5.create_dataset(args.dataset_prefix + 'partition', (len(classes),1), 'i')
print('Writing partitions')
with ProgressBar(max_value=len(classes)) as progress:
    for i in range(len(classes)):
        if classes[i,0] > 0.0:
            if i in active_indices:
                partition[i,0] = 2
            else:
                partition[i,0] = 1
        else:
            if i in inactive_indices:
                partition[i,0] = 2
            else:
                partition[i,0] = 1
        progress.update(i + 1)
data_h5.close()
