import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import LogisticRegression

# set options
logging.basicConfig(format='%(asctime)s %(levelname)s\t%(message)s', level=logging.INFO)

pd.set_option('display.column_space', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 500)

# read in data from CSV
stats_df = pd.read_csv('data/preprocessed.csv')
logging.info('SUCCESSFULLY LOADED CSV INTO DATAFRAME')

# organize features (X) and labels (y)
features = stats_df.drop(['playerID', 'yearID', 'all_star?'], axis=1)
labels = stats_df['all_star?']

# set up logistic regression model
lr = LogisticRegression(class_weight='balanced')
kf = KFold(features.shape[0], random_state=1)

predictions_lr = cross_val_predict(lr, features, labels, cv=kf, verbose=2)
logging.info('SUCCESSFULLY CROSS VALIDATED AND MADE PREDICTIONS ON TEST DATA')

np_predictions_lr = np.asarray(predictions_lr)
np_labels = labels.as_matrix()

print(np_predictions_lr)
print(np_labels)

# statistical analysis
true_positives = np_predictions_lr[(np_predictions_lr == 1) & (np_labels == 1)]
true_negatives = np_predictions_lr[(np_predictions_lr == 0) & (np_labels == 0)]
false_positives = np_predictions_lr[(np_predictions_lr == 1) & (np_labels == 0)]
false_negatives = np_predictions_lr[(np_predictions_lr == 0) & (np_labels == 1)]

precision = len(true_positives) / (len(true_positives) + len(false_positives))
logging.info('PRECISION: %f', precision)
recall = len(true_positives) / (len(true_positives) + len(false_negatives))
logging.info('RECALL: %f', recall)
f1_score = 2 * precision * recall / (precision + recall)
logging.info('F1 SCORE: %f', f1_score)
