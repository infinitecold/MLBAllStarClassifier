import logging
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from team_selection import model_predict

# package options
logging.basicConfig(format='%(levelname)s  %(asctime)s\t%(message)s', level=logging.INFO)

# parameters
input_file_path = '../../data/processed/stats_cv.pickle'
output_file_path = '../../models/random_forest.pickle'

# read in data from pickle
with open(input_file_path, 'rb') as f:
    [X_train, X_test, y_train, y_test] = pickle.load(f)

# prepare identification columns for evaluations
ID_train = X_train[['playerID', 'yearID', 'lgID', 'POS']].copy()
X_train.drop(['playerID', 'yearID', 'lgID', 'POS'], axis=1, inplace=True)
ID_test = X_test[['playerID', 'yearID', 'lgID', 'POS']].copy()
X_test.drop(['playerID', 'yearID', 'lgID', 'POS'], axis=1, inplace=True)

# train model
clf = RandomForestClassifier(n_estimators=400, max_depth=100, max_features=0.1, min_samples_split=5, min_samples_leaf=2, random_state=1)
clf.fit(X_train, y_train)

# test model on train data
train_predictions = model_predict(clf, X_train, ID_train)
train_precision = precision_score(y_train, train_predictions)
train_recall = recall_score(y_train, train_predictions)
train_f1 = f1_score(y_train, train_predictions)

# test model on test data
test_predictions = model_predict(clf, X_test, ID_test)
test_precision = precision_score(y_test, test_predictions)
test_recall = recall_score(y_test, test_predictions)
test_f1 = f1_score(y_test, test_predictions)

logging.info('PRECISION:\ttrain %f  test %f', train_precision, test_precision)
logging.info('RECALL:\t\ttrain %f  test %f', train_recall, test_recall)
logging.info('F1 SCORE:\ttrain %f  test %f', train_f1, test_f1)

# save model to pickle
with open(output_file_path, 'wb') as f:
    pickle.dump(clf, f)
logging.info('SUCCESSFULLY WRITTEN MODEL TO %s', output_file_path)
