import logging
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
from team_selection import model_predict

# package options
logging.basicConfig(format='%(levelname)s  %(asctime)s\t%(message)s', level=logging.INFO)

# read in data from pickle
with open('../../data/processed/stats_cv.pickle', 'rb') as f:
    [X_train, X_test, y_train, y_test] = pickle.load(f)

# prepare identification columns for evaluation
ID_train = X_train[['playerID', 'yearID', 'lgID', 'POS']].copy()
X_train.drop(['playerID', 'yearID', 'lgID', 'POS'], axis=1, inplace=True)
ID_test = X_test[['playerID', 'yearID', 'lgID', 'POS']].copy()
X_test.drop(['playerID', 'yearID', 'lgID', 'POS'], axis=1, inplace=True)

# set up model
clf = RandomForestClassifier(random_state=1)

# manual grid search
grid = ParameterGrid({'n_estimators': [300, 400, 500],
                      'max_features': [0.1, 0.2],
                      'max_depth': [50, 100, 150, 200],
                      'min_samples_split': [5],
                      'min_samples_leaf': [2]})

for param in grid:
    clf.set_params(**param)
    clf.fit(X_train, y_train)
    train_predictions = model_predict(clf, X_train, ID_train)
    train_f1 = f1_score(y_train, train_predictions)
    test_predictions = model_predict(clf, X_test, ID_test)
    test_f1 = f1_score(y_test, test_predictions)
    logging.info('F1 SCORES: train %f test %f WITH PARAMETERS %s', train_f1, test_f1, param)
