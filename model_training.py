import logging
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from team_selection import model_predict

# package options
logging.basicConfig(format='%(asctime)s %(levelname)s\t%(message)s', level=logging.INFO)

# read in data from pickle
with open('data/preproc/stats_cv.pickle', 'rb') as f:
    [X_train, X_test, y_train, y_test] = pickle.load(f)

# prepare identification columns for evaluations
ID_train = X_train[['playerID', 'yearID', 'lgID', 'POS']].copy()
X_train.drop(['playerID', 'yearID', 'lgID', 'POS'], axis=1, inplace=True)
ID_test = X_test[['playerID', 'yearID', 'lgID', 'POS']].copy()
X_test.drop(['playerID', 'yearID', 'lgID', 'POS'], axis=1, inplace=True)

# modify features before training
features_to_drop = ['G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF',
                    'GIDP', 'AVG', 'OBP', 'slugging', 'OPS', 'G_field', 'GS_field', 'InnOuts_field', 'PO_field',
                    'A_field', 'E_field', 'DP_field', 'n_awards']
X_train.drop(features_to_drop, axis=1, inplace=True)
X_test.drop(features_to_drop, axis=1, inplace=True)

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
with open('data/models/random_forest.pickle', 'wb') as f:
    pickle.dump(clf, f)
logging.info('SUCCESSFULLY WRITTEN MODEL TO data/models/random_forest.pickle')
