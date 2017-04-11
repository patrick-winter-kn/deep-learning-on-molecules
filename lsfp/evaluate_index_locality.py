import argparse
from util import neighborhoods


def get_arguments():
    parser = argparse.ArgumentParser(description='Evaluates the locality of the order in the given index')
    parser.add_argument('data', type=str, help='SMILES data used to evauluate the locality')
    parser.add_argument('index', type=str, help='Index file containing the positions for the neighborhoods')
    return parser.parse_args()


args = get_arguments()
out_path = args.index[:-3] + '-eval.h5'
neighborhoods.evaluate_index_locality(args.data, args.index, out_path, 2)
