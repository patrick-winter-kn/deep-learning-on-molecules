from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import numpy


def train(train_data_input, train_data_output, model_path, nr_trees=100):
    random_forest = RandomForestClassifier(n_estimators=nr_trees, min_samples_leaf=10, n_jobs=-1,
                                           class_weight="balanced", verbose=2)
    random_forest.fit(train_data_input, train_data_output)
    joblib.dump(random_forest, model_path)


def predict(test_data_input, model_path):
    random_forest = joblib.load(model_path)
    probabilities = random_forest.predict_proba(test_data_input)
    probabilities = numpy.array(probabilities)[:,:,0].transpose()
    return probabilities