from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import numpy
from progressbar import ProgressBar


def train(train_data_input, train_data_output, model_path, nr_trees=1000):
    random_forest = RandomForestClassifier(n_estimators=nr_trees, min_samples_leaf=10, n_jobs=-1,
                                           class_weight="balanced", verbose=2)
    random_forest.fit(train_data_input, train_data_output)
    joblib.dump(random_forest, model_path)


def predict(test_data_input, model_path):
    random_forest = joblib.load(model_path)
    probabilities = random_forest.predict_proba(test_data_input)
    probabilities = numpy.array(probabilities)
    return probabilities


def numerical_to_classes(numerical_data):
    classes = []
    print('Converting training output')
    with ProgressBar(max_value=len(numerical_data)) as progress:
        for i in range(len(numerical_data)):
            if numerical_data[i][0] >= numerical_data[i][1]:
                classes.append('a')
            else:
                classes.append('i')
            progress.update(i+1)
    return classes
