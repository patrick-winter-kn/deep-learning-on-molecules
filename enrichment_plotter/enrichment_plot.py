import argparse
import h5py
from plotters import enrichment_plotter


def get_arguments():
    parser = argparse.ArgumentParser(description='Creates an enrichment plot for the given predictions')
    parser.add_argument('predictions_files', type=str, nargs='+', help='File containing the predictions')
    parser.add_argument('--predictions_names', type=str, nargs='+', default=[],
                        help='Names for the predictions used in the legend (default: Prediction 1, Prediction 2, ...)')
    parser.add_argument('--classes_file', type=str, default=None,
                        help='File containing the actual classes (default: Use first predictions file)')
    parser.add_argument('--enrichment_plot_file', type=str, default=None,
                        help='The enrichment plot file (default: <path-of-first-predictions-file>/enrichmentplot.svg)')
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
    args.classes_file = args.predictions_files[0]
if args.show:
    args.enrichment_plot_file = None
else:
    if not args.enrichment_plot_file:
        args.enrichment_plot_file = args.predictions_files[0][:args.predictions_files[0].rfind('/') + 1]\
                                    + 'enrichmentplot.svg'
for i in range(len(args.predictions_names), len(args.predictions_files)):
    args.predictions_names.append('Prediction ' + str(i))
predictions_h5s = []
predictions = []
for predictions_file in args.predictions_files:
    predictions_h5 = h5py.File(predictions_file, 'r')
    prediction = predictions_h5[args.predictions_data_set_name]
    predictions_h5s.append(predictions_h5)
    predictions.append(prediction)
classes_h5 = h5py.File(args.classes_file, 'r')
classes = classes_h5[args.classes_data_set_name]
enrichment_plotter.plot(predictions, args.predictions_names, classes, args.enrichment_factors,
                        args.enrichment_plot_file, args.show)
classes_h5.close()
for predictions_h5 in predictions_h5s:
    predictions_h5.close()
