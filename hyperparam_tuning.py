import logging
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
from team_selection import model_predict

# package options
logging.basicConfig(format='%(asctime)s %(levelname)s\t%(message)s', level=logging.INFO)

# parameters
repetitions = 5

# read in data from pickle
with open('data/preproc/stats_cv.pickle', 'rb') as f:
    [X_train, X_test, y_train, y_test] = pickle.load(f)

# drop identification columns
ID_train = X_train[['playerID', 'yearID', 'lgID', 'POS']].copy()
X_train.drop(['playerID', 'yearID', 'lgID', 'POS'], axis=1, inplace=True)
ID_test = X_test[['playerID', 'yearID', 'lgID', 'POS']].copy()
X_test.drop(['playerID', 'yearID', 'lgID', 'POS'], axis=1, inplace=True)

# set up model
clf = RandomForestClassifier(random_state=1)

# manual grid search
grid = ParameterGrid({'n_estimators': [100, 200, 300, 400, 500],
                      'criterion': ['gini', 'entropy'],
                      'max_features': [0.2, 'auto', 0.8],
                      'max_depth': [50, 100, 150, 200],
                      'min_samples_split': [1, 2, 5, 10, 15],
                      'min_samples_leaf': [1, 2, 5, 10, 15]})

for param in grid:
    clf.set_params(**param)
    clf.fit(X_train, y_train)
    train_predictions = model_predict(clf, X_train, ID_train)
    train_f1 = f1_score(y_train, train_predictions)
    test_predictions = model_predict(clf, X_test, ID_test)
    test_f1 = f1_score(y_test, test_predictions)
    logging.info('F1 SCORES: train %f test %f WITH PARAMETERS %s', train_f1, test_f1, param)
