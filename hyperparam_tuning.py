import logging
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid

# package options
logging.basicConfig(format='%(asctime)s %(levelname)s\t%(message)s', level=logging.INFO)

# read in data from pickle
with open('data/preproc/stats_train_test.pickle', 'rb') as f:
    [X_train, X_test, y_train, y_test] = pickle.load(f)

# copy all-star data in training set
copies = 5
train = pd.concat([X_train, y_train], axis=1)
train = train.append([train[train['all_star?'] == 1]] * copies, ignore_index=True)
X_train = train.drop(['all_star?'], axis=1)
y_train = train['all_star?']

# set up model
clf = RandomForestClassifier()

# manual grid search to avoid K-Fold cross-validation in sklearn.model_selection.GridSearchCV
grid = ParameterGrid({'n_estimators': [400],
                      'max_features': [0.2],
                      'max_depth': [120],
                      'min_samples_split': [15],  # larger -> weaker model
                      'min_samples_leaf': [2]})  # larger -> weaker model

for param in grid:
    clf.set_params(**param)
    clf.fit(X_train, y_train)
    train_predictions = clf.predict(X_train)
    train_f1 = f1_score(y_train, train_predictions)
    test_predictions = clf.predict(X_test)
    test_f1 = f1_score(y_test, test_predictions)
    logging.info('F1 SCORES: train %f test %f WITH PARAMETERS %s', train_f1, test_f1, param)
