import logging
import pandas as pd

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
