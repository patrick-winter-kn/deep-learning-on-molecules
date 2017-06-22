import numpy
import math
from progressbar import ProgressBar


def calculate_stats(predictions_list, classes, enrichment_factors):
    positives = positives_count(classes)
    actives_list = []
    efs_list = []
    auc_list = []
    for i in range(len(predictions_list)):
        print('Calculating enrichment stats')
        predictions = predictions_list[i]
        actives, efs, auc = enrichment_stats(predictions[:, 0].argsort()[::-1], enrichment_factors, classes, positives)
        actives_list.append(actives)
        efs_list.append(efs)
        auc_list.append(auc)
        # diversity = diversity_ratio(predictions)
        # print('Diversity ratio: ' + str(diversity))
    return efs_list, auc_list


def diversity_ratio(predictions):
    print('Calculating diversity')
    results = set()
    with ProgressBar(max_value=len(predictions)) as progress:
        for i in range(len(predictions)):
            results.add(predictions[i][0])
            progress.update(i+1)
    return len(results)/len(predictions)


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
