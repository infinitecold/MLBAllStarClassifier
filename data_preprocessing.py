import logging
import pandas as pd

# package options
logging.basicConfig(format='%(asctime)s %(levelname)s\t%(message)s', level=logging.INFO)

pd.set_option('display.column_space', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 500)

# parameters
column_keys = ['playerID', 'yearID']
batting_stats = ['G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF', 'GIDP']
fielding_stats = ['G', 'GS', 'InnOuts', 'PO', 'A', 'E', 'DP']
position_columns = ['G_p', 'G_c', 'G_1b', 'G_2b', 'G_3b', 'G_ss', 'G_lf', 'G_cf', 'G_rf', 'G_dh']
positions_to_remove = ['P']
eras = {'1933-1941': (1933, 1941), '1942-1945': (1942, 1945), '1946-1962': (1946, 1962), '1963-1976': (1963, 1976),
        '1977-1992': (1977, 1992), '1993-2009': (1993, 2009), '2010-': (2010, 2100)}

# read in data from CSVs
batting_df = pd.read_csv('data/raw/Batting.csv', usecols=column_keys+batting_stats)
fielding_df = pd.read_csv('data/raw/Fielding.csv', usecols=column_keys+fielding_stats)
appearances_df = pd.read_csv('data/raw/Appearances.csv')
awards_df = pd.read_csv('data/raw/AwardsPlayers.csv', usecols=['playerID', 'awardID', 'yearID'])
allstar_df = pd.read_csv('data/raw/AllstarFull.csv', usecols=['playerID', 'yearID'])
logging.info('SUCCESSFULLY LOADED CSV(\'s) INTO DATAFRAME(\'s)')

# PROCESSING STATS FROM DATA
# 1. batting
batting_df = batting_df.groupby(column_keys, as_index=False).sum()

stats_df = batting_df
logging.info('SUCCESSFULLY ADDED BATTING STATS (%d)', len(batting_df))

# 2. fielding
fielding_df.columns = column_keys + [str(col) + '_field' for col in fielding_df.columns if str(col) in fielding_stats]
fielding_df = fielding_df.groupby(column_keys, as_index=False).sum()

stats_df = pd.merge(stats_df, fielding_df, how='outer', on=column_keys)
logging.info('SUCCESSFULLY ADDED FIELDING STATS (%d)', len(fielding_df))

# 3. appearances - used to choose POS
appearances_df = appearances_df.groupby(column_keys, as_index=False).sum()

def choose_position(row):
    row = pd.to_numeric(row, errors='coerce')
    best_position_label = row.reindex(position_columns).idxmax()  # e.g. G_3b
    best_position = best_position_label[2:].upper()  # e.g. 3B
    # if player is not primarily playing a position to be removed, find next position not to be removed
    if best_position in positions_to_remove and float(row[best_position_label]) / row['G_all'] < 0.8:
        while best_position in positions_to_remove:
            row.drop(best_position_label, inplace=True)
            best_position_label = row.reindex(position_columns).idxmax()
            best_position = best_position_label[2:].upper()
    return best_position

appearances_df['POS'] = appearances_df.apply(choose_position, axis=1)
appearances_df = appearances_df[column_keys + ['POS']]
stats_df = pd.merge(stats_df, appearances_df, how='outer', on=column_keys)
logging.info('SUCCESSFULLY ADDED POSITION (POS) STAT (%d)', len(appearances_df))

logging.info('TOTAL PLAYER SEASONS FROM DATA: %d', len(stats_df))

# drop players with positions to be removed
stats_df = stats_df[~stats_df['POS'].isin(positions_to_remove)]
logging.info('TOTAL PLAYER SEASONS WITHOUT %s POSITIONS: %d', positions_to_remove, len(stats_df))

# drop players before 1933 (first all-star game in 1933)
stats_df = stats_df[stats_df['yearID'] >= 1933]
logging.info('TOTAL PLAYER SEASONS 1933 OR LATER: %d', len(stats_df))

# 4. number of awards (at the time of the season)
stats_df['n_awards'] = 0
for _, row in awards_df.iterrows():
    stats_df.loc[(stats_df['playerID'] == row['playerID']) & (stats_df['yearID'] > row['yearID']), ['n_awards']] += 1
logging.info('SUCCESSFULLY ADDED AWARDS STATS (ALL)')

# 5. eras
for era_label in eras:
    stats_df[era_label] = 0
for label, era in eras.iteritems():
    stats_df.loc[(stats_df['yearID'] >= era[0]) & (stats_df['yearID'] <= era[1]), [label]] = 1
logging.info('SUCCESSFULLY ADDED ERA STATS (ALL)')

# 6. all-star appearances
stats_df['all_star?'] = 0
for _, row in allstar_df.iterrows():
    stats_df.loc[(stats_df['playerID'] == row['playerID']) & (stats_df['yearID'] == row['yearID']), ['all_star?']] = 1
logging.info('SUCCESSFULLY ADDED ALL-STAR STATS (%s)', len(allstar_df))

# FILLING MISSING DATA
columns_missing_data = stats_df.columns[stats_df.isnull().any()].tolist()
logging.info('COLUMNS IN stats_df MISSING DATA: %s', columns_missing_data)

for column in columns_missing_data:
    allstar_avg = stats_df[stats_df['all_star?'] == 1][column].mean()
    stats_df.loc[(stats_df[column].isnull()) & (stats_df['all_star?'] == 1), [column]] = allstar_avg
    non_allstar_avg = stats_df[stats_df['all_star?'] == 0][column].mean()
    stats_df.loc[(stats_df[column].isnull()) & (stats_df['all_star?'] == 0), [column]] = non_allstar_avg
    logging.info('COLUMN %s AVERAGES: ALL-STAR %f NON-ALL-STAR %f', column, allstar_avg, non_allstar_avg)

# ADDING STATS USING EXISTING DATA
# i. batting average
stats_df['AVG'] = stats_df['H'] / stats_df['AB']

# ii. on-base percentage
PA_no_SH = stats_df['AB'] + stats_df['BB'] + stats_df['HBP'] + stats_df['SF']  # plate appearances without SH
stats_df['OBP'] = (stats_df['H'] + stats_df['BB'] + stats_df['HBP']) / PA_no_SH

# iii. slugging percentage
singles = stats_df['H'] - stats_df['2B'] - stats_df['3B'] - stats_df['HR']
stats_df['slugging'] = (stats_df['HR'] * 4 + stats_df['3B'] * 3 + stats_df['2B'] * 2 + singles) / stats_df['AB']

# replace NaN values (players with 0 AB's) with 0's
stats_df.fillna(0, inplace=True)

# iv. on-base plus slugging
stats_df['OPS'] = stats_df['OBP'] + stats_df['slugging']

# re-order columns
stats_df = stats_df[['playerID', 'yearID', 'POS', 'G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO',
                     'IBB', 'HBP', 'SH', 'SF', 'GIDP', 'AVG', 'OBP', 'slugging', 'OPS', 'G_field', 'GS_field',
                     'InnOuts_field', 'PO_field', 'A_field', 'E_field', 'DP_field', 'n_awards', '1933-1941',
                     '1942-1945', '1946-1962', '1963-1976', '1977-1992', '1993-2009', '2010-', 'all_star?']]

# save data to CSV
stats_df.to_csv('data/preproc/stats.csv', index=False)
logging.info('SUCCESSFULLY WRITTEN DATAFRAME stats_df TO data/preproc/stats.csv')
