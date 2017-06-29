import numpy
import math
from progressbar import ProgressBar
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot


def plot(predictions_list, prediction_names, classes, enrichment_factors, enrichment_plot_file):
    positives = positives_count(classes)
    actives_list = []
    efs_list = []
    auc_list = []
    for i in range(len(predictions_list)):
        print('Calculating stats for ' + prediction_names[i])
        predictions = predictions_list[i]
        actives, efs, auc = enrichment_stats(predictions[:, 0].argsort()[::-1], enrichment_factors, classes, positives)
        actives_list.append(actives)
        efs_list.append(efs)
        auc_list.append(auc)
    axis = pyplot.subplots()[1]
    axis.grid(True, linestyle='--')
    # Plot random line
    pyplot.plot((0,len(actives)), (0,actives[-1]), ls='-', c='0.75', label='Random')
    # Plot actives
    for i in range(len(predictions_list)):
        pyplot.plot(actives_list[i], label=prediction_names[i]+' (AUC: ' + str(round(auc_list[i], 2)) + ')')
    # Add enrichment factors
    for percent in sorted(enrichment_factors):
        x_start_end = int(math.ceil(percent * 0.01 * len(classes)))
        y_end = 0
        for actives in actives_list:
            y_end = max(y_end, actives[x_start_end])
        ef_label = 'Enrichment factor ' + str(percent) + '%'
        for i in range(len(efs_list)):
            ef_label += '\n' + prediction_names[i] + ': ' + str(round(efs_list[i][percent], 2))
        pyplot.plot((x_start_end, x_start_end), (0, y_end), ls='--', label=ef_label)
    pyplot.ylabel('Active Compounds')
    pyplot.xlabel('Compounds')
    pyplot.legend(loc='lower right', fancybox=True)
    pyplot.tight_layout()
    if enrichment_plot_file:
        pyplot.savefig(enrichment_plot_file, format='svg', transparent=True)
    else:
        pyplot.show()
    pyplot.close('all')
    return auc_list, efs_list


def positives_count(classes):
    positives = 0
    print('Counting actives')
    with ProgressBar(max_value=len(classes)) as progress:
        i = 0
        for row in classes:
            if numpy.where(row == max(row))[0] == 0:
                positives += 1
            i += 1
            progress.update(i)
    print('Found ' + str(positives) + ' actives and ' + str(len(classes) - positives) + ' inactives')
    return positives


def enrichment_stats(indices, ef_percent, classes, positives):
    actives = [0]
    # efs maps the percent to the number of found positives
    efs = {}
    for percent in ef_percent:
        efs[percent] = 0
    found = 0
    curve_sum = 0
    with ProgressBar(max_value=len(indices)) as progress:
        for i in range(len(indices)):
            row = classes[indices[i]]
            # Check if index (numpy.where) of maximum value (max(row)) in row is 0 (==0)
            # This means the active value is higher than the inactive value
            if numpy.where(row == max(row))[0] == 0:
                found += 1
                for percent in efs.keys():
                    # If i is still part of the fraction count the number of founds up
                    if i < int(math.floor(len(indices)*(percent*0.01))):
                        efs[percent] += 1
            curve_sum += found
            actives.append(found)
            progress.update(i + 1)
    # AUC = sum of found positives for every x / (positives * (number of samples + 1))
    # + 1 is added to the number of samples for the start with 0 samples selected
    auc = curve_sum / (positives * (len(classes) + 1))
    print('AUC: ' + str(auc))
    # Turn number of found positives into enrichment factor by dividing the number of positives found at random
    for percent in sorted(efs.keys()):
        efs[percent] /= (positives * (percent * 0.01))
        print('EF at ' + str(percent) + '%: ' + str(efs[percent]))
    return actives, efs, auc
