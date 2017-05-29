import argparse
import h5py
from plotters import enrichment_plotter


def get_arguments():
    parser = argparse.ArgumentParser(description='Creates an enrichment plot for the given predictions')
    parser.add_argument('predictions_file', type=str, help='File containing the predictions')
    parser.add_argument('--classes_file', type=str, default=None,
                        help='File containing the actual classes (default: <predictions_file>)')
    parser.add_argument('--enrichment_plot_file', type=str, default=None,
                        help='The enrichment plot file (default: <predictions_file>-enrichmentplot.svg)')
    parser.add_argument('--predictions_data_set_name', type=str, default='predictions',
                        help='Name of the data set containing the predictions (default: predictions)')
    parser.add_argument('--classes_data_set_name', type=str, default='classes',
                        help='Name of the data set containing the classes (default: classes)')
    parser.add_argument('--enrichment_factors', type=int, default=[], nargs='+',
                        help='Print enrichment factor for the given percent (default: None)')
    parser.add_argument('--show', action='store_true',
                        help='Shows the plot in a window instead of saving it (default: False)')
    return parser.parse_args()


args = get_arguments()
if not args.classes_file:
    args.classes_file = args.predictions_file
if not args.enrichment_plot_file:
    args.enrichment_plot_file = args.predictions_file[:args.predictions_file.rfind('.')] + '-enrichmentplot.svg'
predictions_h5 = h5py.File(args.predictions_file, 'r')
classes_h5 = h5py.File(args.classes_file, 'r')
predictions = predictions_h5[args.predictions_data_set_name]
classes = classes_h5[args.classes_data_set_name]
enrichment_plotter.plot(predictions, classes, args.enrichment_factors, args.enrichment_plot_file, args.show)
predictions_h5.close()
classes_h5.close()
