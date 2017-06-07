from progressbar import ProgressBar
import numpy


def count(classes):
    actives = 0
    print('Counting active compounds')
    with ProgressBar(max_value=len(classes)) as progress:
        i = 0
        for row in classes:
            if numpy.where(row == max(row))[0] == 0:
                actives += 1
            i += 1
            progress.update(i)
    print('Found ' + str(actives) + ' active compounds')
    return actives
