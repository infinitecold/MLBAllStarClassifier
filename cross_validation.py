import logging
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

logging.basicConfig(format='%(asctime)s %(levelname)s\t%(message)s', level=logging.INFO)

# read in data from CSV
stats_df = pd.read_csv('preprocessed/stats.csv')
logging.info('SUCCESSFULLY LOADED CSV INTO DATAFRAME')

# organize features (X) and labels (y)
X = stats_df.drop(['playerID', 'yearID', 'all_star?'], axis=1)
y = stats_df['all_star?']

# cross-validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
logging.info('SUCCESSFULLY PREPARED TRAINING AND TESTING SETS')

# save data to pickle
with open('preprocessed/stats_train_test.pickle', 'wb') as f:
    pickle.dump([X_train, X_test, y_train, y_test], f)
logging.info('SUCCESSFULLY WRITTEN TRAINING AND TESTING SETS TO preprocessed/stats_train_test.pickle')
