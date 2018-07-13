import logging
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

# package options
logging.basicConfig(format='%(asctime)s %(levelname)s\t%(message)s', level=logging.INFO)

# read in data from pickle
with open('data/preproc/stats_cv.pickle', 'rb') as f:
    [X_train, X_test, y_train, y_test] = pickle.load(f)

# drop identification columns
X_train.drop(['playerID', 'yearID', 'lgID', 'POS'], axis=1, inplace=True)
X_test.drop(['playerID', 'yearID', 'lgID', 'POS'], axis=1, inplace=True)

# make all-star data copies in training set
# copies = 5
# train = pd.concat([X_train, y_train], axis=1)
# train = train.append([train[train['all_star?'] == 1]] * copies, ignore_index=True)
# X_train = train.drop(['all_star?'], axis=1)
# y_train = train['all_star?']

# modify features before training
features_to_drop = ['G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF',
                    'GIDP', 'AVG', 'OBP', 'slugging', 'OPS', 'G_field', 'GS_field', 'InnOuts_field', 'PO_field',
                    'A_field', 'E_field', 'DP_field', 'n_awards']
X_train.drop(features_to_drop, axis=1, inplace=True)
X_test.drop(features_to_drop, axis=1, inplace=True)

# train model
clf = RandomForestClassifier(n_estimators=200, max_depth=200, min_samples_split=5, min_samples_leaf=3, random_state=1)
clf.fit(X_train, y_train)

# test model on train data
train_predictions = clf.predict(X_train)
train_precision = precision_score(y_train, train_predictions)
train_recall = recall_score(y_train, train_predictions)
train_f1 = f1_score(y_train, train_predictions)

# test model on test data
test_predictions = clf.predict(X_test)
test_precision = precision_score(y_test, test_predictions)
test_recall = recall_score(y_test, test_predictions)
test_f1 = f1_score(y_test, test_predictions)

# test model on test data with adjustable threshold
probability_threshold = 0.35
probabilities = clf.predict_proba(X_test)
probabilities_results = np.append(probabilities, y_test.values.reshape(len(y_test), 1), axis=1)
true_positives = true_negatives = false_positives = false_negatives = 0.0
for result in probabilities_results:
    if result[1] > probability_threshold and result[2] == 1:
        true_positives += 1
    elif result[1] <= probability_threshold and result[2] == 0:
        true_negatives += 1
    elif result[1] > probability_threshold and result[2] == 0:
        false_positives += 1
    elif result[1] <= probability_threshold and result[2] == 1:
        false_negatives += 1

test_threshold_precision = true_positives / (true_positives + false_positives)
test_threshold_recall = true_positives / (true_positives + false_negatives)
test_threshold_f1 = 2 * test_threshold_precision * test_threshold_recall / (test_threshold_precision + test_threshold_recall)

logging.info('TRAIN TEAM SIZE:\tPREDICT %d\tACTUAL %d', len(train_predictions[train_predictions == 1]),
             len(y_train[y_train == 1]))
logging.info('TEST TEAM SIZE:\t\tPREDICT %d\t\tACTUAL %d', len(test_predictions[test_predictions == 1]),
             len(y_test[y_test == 1]))

logging.info('PRECISION:\ttrain %.3f  test (0.50) %.3f  test (%.2f) %.3f', train_precision, test_precision,
             probability_threshold, test_threshold_precision)
logging.info('RECALL:\t\ttrain %.3f  test (0.50) %.3f  test (%.2f) %.3f', train_recall, test_recall,
             probability_threshold, test_threshold_recall)
logging.info('F1 SCORE:\ttrain %.3f  test (0.50) %.3f  test (%.2f) %.3f', train_f1, test_f1, probability_threshold,
             test_threshold_f1)

# save model to pickle
with open('data/models/random_forest.pickle', 'wb') as f:
    pickle.dump(clf, f)
logging.info('SUCCESSFULLY WRITTEN MODEL TO data/models/random_forest.pickle')
