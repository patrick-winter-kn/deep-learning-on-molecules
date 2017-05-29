import numpy
import math
from progressbar import ProgressBar
from matplotlib import pyplot


def plot(predictions, classes, enrichment_factors, enrichment_plot_file, show):
    indices = predictions[:, 0].argsort()[::-1]
    actives, efs, auc = enrichment_stats(indices, enrichment_factors, classes)
    axis = pyplot.subplots()[1]
    axis.grid(True, linestyle='--')
    # Plot random line
    pyplot.plot((0,len(actives)), (0,actives[-1]), ls='-', c='0.75', label='Random')
    # Plot actives
    pyplot.plot(actives, label='Predicted (AUC: ' + str(round(auc, 2)) + ')')
    # Add enrichment factors
    for percent in sorted(efs):
        x_start_end = int(math.ceil(percent * 0.01 * len(classes)))
        y_end = actives[x_start_end]
        pyplot.plot((x_start_end, x_start_end), (0, y_end), ls='--',
                    label='Enrichment factor ' + str(percent) + '%: ' + str(round(efs[percent], 2)))
    pyplot.ylabel('Active Compounds')
    pyplot.xlabel('Compounds')
    pyplot.legend(loc='lower right')
    pyplot.tight_layout()
    if show:
        pyplot.show()
    else:
        pyplot.savefig(enrichment_plot_file, format='svg', transparent=True)


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


def enrichment_stats(indices, ef_percent, classes):
    actives = [0]
    positives = positives_count(classes)
    # efs maps the percent to the number of found positives
    efs = {}
    for percent in ef_percent:
        efs[percent] = 0
    found = 0
    curve_sum = 0
    print('Calculating enrichment stats')
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
    for percent in efs.keys():
        efs[percent] /= (positives * (percent * 0.01))
        print('EF at ' + str(percent) + '%: ' + str(efs[percent]))
    return actives, efs, auc
