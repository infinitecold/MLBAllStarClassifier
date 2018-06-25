import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# set options
logging.basicConfig(format='%(asctime)s %(levelname)s\t%(message)s', level=logging.INFO)

np.set_printoptions(threshold=np.nan)

pd.set_option('display.column_space', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 500)

# read in data from CSV
stats_df = pd.read_csv('preprocessed/stats.csv')
logging.info('SUCCESSFULLY LOADED CSV INTO DATAFRAME')

# organize features (X) and labels (y)
X = stats_df.drop(['playerID', 'yearID', 'all_star?'], axis=1)
y = stats_df['all_star?']

# cross-validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
logging.info('SUCCESSFULLY PREPARED TRAINING AND TESTING SETS')

# set up model
clf = RandomForestClassifier(random_state=1, n_estimators=12, max_depth=11, min_samples_leaf=1, class_weight={0: 1, 1: 3})
# clf = MLPClassifier(hidden_layer_sizes=(800, 400, 100))

# train model
clf.fit(X_train, y_train)

# test model
predictions = clf.predict(X_test)
np_predictions = np.asarray(predictions)
np_labels = y_test

# statistical analysis
true_positives = np_predictions[(np_predictions == 1) & (np_labels == 1)]
true_negatives = np_predictions[(np_predictions == 0) & (np_labels == 0)]
false_positives = np_predictions[(np_predictions == 1) & (np_labels == 0)]
false_negatives = np_predictions[(np_predictions == 0) & (np_labels == 1)]

precision = precision_score(np_labels, np_predictions)
logging.info('PRECISION: %.3f', precision)
recall = recall_score(np_labels, np_predictions)
logging.info('RECALL: %.3f', recall)
f1 = f1_score(np_labels, np_predictions)
logging.info('F1 SCORE: %.3f', f1)
