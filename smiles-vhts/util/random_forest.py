from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


def train(train_data_input, train_data_output, model_path, nr_trees=10000):
    random_forest = RandomForestClassifier(n_estimators=nr_trees, min_samples_leaf=10, class_weight="balanced", n_jobs=-1)
    random_forest.fit(train_data_input, train_data_output)
    joblib.dump(random_forest, model_path)


def predict(test_data_input, model_path):
    random_forest = joblib.load(model_path)
    probabilities = zip(*random_forest.predict_proba(test_data_input))
    return probabilities
