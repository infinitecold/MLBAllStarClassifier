import logging
import pandas as pd
import pickle

# package options
logging.basicConfig(format='%(asctime)s %(levelname)s\t%(message)s', level=logging.INFO)

# read in data from CSV
stats_df = pd.read_csv('data/preproc/stats.csv')

# cross-validation by year
train_df = stats_df[stats_df['yearID'] < 2010].copy()
test_df = stats_df[stats_df['yearID'] >= 2010].copy()

# fill missing data with all-star/non-all-star averages from train set
columns_missing_data = train_df.columns[train_df.isnull().any()].tolist()
logging.info('COLUMNS MISSING DATA: %s', columns_missing_data)
for column in columns_missing_data:
    allstar_avg = train_df[train_df['all_star?'] == 1][column].mean()
    train_df.loc[(train_df[column].isnull()) & (train_df['all_star?'] == 1), [column]] = allstar_avg
    test_df.loc[(test_df[column].isnull()) & (test_df['all_star?'] == 1), [column]] = allstar_avg
    non_allstar_avg = train_df[train_df['all_star?'] == 0][column].mean()
    train_df.loc[(train_df[column].isnull()) & (train_df['all_star?'] == 0), [column]] = non_allstar_avg
    test_df.loc[(test_df[column].isnull()) & (test_df['all_star?'] == 0), [column]] = non_allstar_avg
    logging.info('TRAIN SET COLUMN %s AVERAGES: ALL-STAR %f NON-ALL-STAR %f', column, allstar_avg, non_allstar_avg)

# organize features (X) and labels (y)
X_train = train_df.drop(['all_star?'], axis=1)
X_test = test_df.drop(['all_star?'], axis=1)
y_train = train_df['all_star?']
y_test = test_df['all_star?']

# save data to pickle
with open('data/preproc/stats_cv.pickle', 'wb') as f:
    pickle.dump([X_train, X_test, y_train, y_test], f)
logging.info('SUCCESSFULLY WRITTEN TRAINING AND TESTING SETS TO data/preproc/stats_cv.pickle')
