import h5py
import argparse
import os
from rdkit import Chem
from rdkit.Chem import Draw
from progressbar import ProgressBar
from util import partition_ref, oversample_ref, shuffle
import threading
import math
import multiprocessing
from rdkit.Chem.Draw.MolDrawing import DrawingOptions


def get_arguments():
    parser = argparse.ArgumentParser(description='Preprocess data for image based learning')
    parser.add_argument('data', type=str, help='Input data file containing the smiles, the classes and the partitions')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale of the rendered images (default: 1.0)')
    return parser.parse_args()


class ImageRenderer(threading.Thread):

    def __init__(self, smiles, offset, directory, progress, scale):
        threading.Thread.__init__(self)
        self.smiles = smiles
        self.offset = offset
        self.directory = directory
        self.progress = progress
        self.scale = scale

    def run(self):
        drawingSize = (800 * self.scale, 800 * self.scale)
        options = DrawingOptions()
        options.dotsPerAngstrom = 30 * self.scale
        for i in range(len(self.smiles)):
            image_path = self.directory + str(self.offset + i) + '.png'
            if not os.path.exists(image_path):
                molecule = Chem.MolFromSmiles(self.smiles[i])
                Draw.MolToFile(molecule, image_path, size=drawingSize, options=options)
            self.progress.increment()


class ProgressMonitor:

    def __init__(self, max_value):
        self.max_value = max_value
        self.n = 0
        self.lock = threading.Lock()
        self.bar = ProgressBar(max_value=max_value)

    def increment(self):
        self.lock.acquire()
        self.n += 1
        self.bar.update(self.n)
        if self.n == self.max_value:
            self.bar.finish()
        self.lock.release()


args = get_arguments()
data_h5 = h5py.File(args.data, 'r')
smiles = data_h5['smiles']
prefix = args.data[:args.data.rfind('.')]
image_dir = args.data[:args.data.rfind('/')] + '/images/'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

nr_threads = multiprocessing.cpu_count() * 2
print('Rendering images')
progress = ProgressMonitor(len(smiles))
threads = []
smiles_per_thread = math.ceil(len(smiles)/nr_threads)
for i in range(nr_threads):
    start = i * smiles_per_thread
    end = (i + 1) * smiles_per_thread
    thread = ImageRenderer(smiles[start:end], start, image_dir, progress, args.scale)
    thread.start()
    threads.append(thread)
for i in range(len(threads)):
    threads[i].join()
print('Finished rendering images')

if not os.path.exists(prefix + '-train.h5') or not os.path.exists(prefix + '-validate.h5'):
    partition_ref.write_partitions(args.data, {1: 'train', 2: 'validate'})
    oversample_ref.oversample(prefix + '-train.h5', args.data)
    shuffle.shuffle(prefix + '-train.h5')

data_h5.close()
