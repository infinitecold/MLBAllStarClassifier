import logging
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# package options
logging.basicConfig(format='%(asctime)s %(levelname)s\t%(message)s', level=logging.INFO)

# read in data from pickle
with open('preprocessed/stats_train_test.pickle', 'rb') as f:
    [X_train, X_test, y_train, y_test] = pickle.load(f)

# set up model
clf = RandomForestClassifier()

# search through grid for best parameters
grid = {'n_estimators': [10, 20, 50, 100, 200],
        'max_features': ['auto', 'log2'],
        'max_depth': [20, 50, 100, 200],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10, 20],
        'bootstrap': [True, False]}
clf_search = RandomizedSearchCV(clf, grid, n_iter=25, scoring='f1', cv=5, random_state=1, n_jobs=-1)

clf_search.fit(X_train, y_train)

logging.info('BEST PARAMETERS FOR RANDOM FOREST CLASSIFIER: %s', clf_search.best_params_)
