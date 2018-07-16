import logging
import pandas as pd
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score

# package options
logging.basicConfig(format='%(asctime)s %(levelname)s\t%(message)s', level=logging.INFO)

# parameters
leagues = ['AL', 'NL']
positions = ['C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF']

# read in data from CSVs
team_sizes_df = pd.read_csv('data/preproc/team_sizes.csv')

# for each league of each year, select all-stars in league_year_stats and then update predictions in probabilities_df
def model_predict(clf, X, ID):
    probabilities = clf.predict_proba(X)
    probabilities_df = pd.concat([ID.reset_index(drop=True), pd.DataFrame(probabilities)], axis=1)
    probabilities_df['prediction'] = 0
    for year in sorted(probabilities_df['yearID'].unique()):
        team_sizes = team_sizes_df[team_sizes_df['yearID'] == year]
        for league in leagues:
            league_year_stats = probabilities_df[(probabilities_df['yearID'] == year) & (probabilities_df['lgID'] == league)].copy().reset_index(drop=True)
            league_count = team_sizes[league].iloc[0]
            # first fill starting positions
            for position in positions:
                allstar_index = league_year_stats[league_year_stats['POS'] == position][1].idxmax()
                allstar = league_year_stats.iloc[allstar_index]['playerID']
                probabilities_df.loc[(probabilities_df['playerID'] == allstar) & (probabilities_df['yearID'] == year), ['prediction']] = 1
                league_year_stats.drop(allstar_index, inplace=True)
                league_year_stats.reset_index(drop=True, inplace=True)
                league_count -= 1
                logging.debug('PREDICTED %d %s ALL-STAR: PLAYER %s PLAYING POSITION %s', year, league, allstar, position)
            # then fill reserves
            reserves = league_year_stats.nlargest(league_count, [1])
            for _, reserve in reserves.iterrows():
                probabilities_df.loc[(probabilities_df['playerID'] == reserve['playerID']) & (probabilities_df['yearID'] == year), ['prediction']] = 1
                logging.debug('PREDICTED %d %s ALL-STAR: PLAYER %s AS RESERVE', year, league, reserve['playerID'])

    return probabilities_df['prediction'].values.flatten()

if __name__ == '__main__':
    # read in data and model from pickle
    with open('data/preproc/stats_cv.pickle', 'rb') as f:
        [X_train, X_test, y_train, y_test] = pickle.load(f)

    with open('data/models/random_forest.pickle', 'rb') as f:
        clf = pickle.load(f)

    # prepare identification columns for evaluation
    ID_train = X_train[['playerID', 'yearID', 'lgID', 'POS']].copy()
    X_train.drop(['playerID', 'yearID', 'lgID', 'POS'], axis=1, inplace=True)
    ID_test = X_test[['playerID', 'yearID', 'lgID', 'POS']].copy()
    X_test.drop(['playerID', 'yearID', 'lgID', 'POS'], axis=1, inplace=True)

    # modify features before predicting
    features_to_drop = ['G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF',
                        'GIDP', 'AVG', 'OBP', 'slugging', 'OPS', 'G_field', 'GS_field', 'InnOuts_field', 'PO_field',
                        'A_field', 'E_field', 'DP_field', 'n_awards']
    X_train.drop(features_to_drop, axis=1, inplace=True)
    X_test.drop(features_to_drop, axis=1, inplace=True)

    train_predictions = model_predict(clf, X_train, ID_train)
    train_precision = precision_score(y_train, train_predictions)
    train_recall = recall_score(y_train, train_predictions)
    train_f1 = f1_score(y_train, train_predictions)

    test_predictions = model_predict(clf, X_test, ID_test)
    test_precision = precision_score(y_test, test_predictions)
    test_recall = recall_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions)

    logging.info('PRECISION:\ttrain %.3f  test %.3f', train_precision, test_precision)
    logging.info('RECALL:\t\ttrain %.3f  test %.3f', train_recall, test_recall)
    logging.info('F1 SCORE:\ttrain %.3f  test %.3f', train_f1, test_f1)
