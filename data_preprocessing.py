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

first_allstar_year = 1933
current_year = 2018
eras = {'1933-1941': (1933, 1941), '1942-1945': (1942, 1945), '1946-1962': (1946, 1962), '1963-1976': (1963, 1976),
        '1977-1992': (1977, 1992), '1993-2009': (1993, 2009), '2010-': (2010, current_year)}

games_threshold = 100
diff_percentage = 0.1
pos_stats_to_diff = ['G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'BB', 'IBB', 'HBP', 'SH', 'SF', 'AVG', 'OBP',
                     'slugging', 'OPS', 'G_field', 'GS_field', 'InnOuts_field', 'PO_field', 'A_field', 'DP_field',
                     'n_awards']  # the higher the stat the better
neg_stats_to_diff = ['CS', 'SO', 'GIDP', 'E_field']  # the lower the stat the better

all_stats_ordered = ['playerID', 'yearID', 'POS', 'G', 'G_diff', 'AB', 'AB_diff', 'R', 'R_diff', 'H', 'H_diff', '2B',
                     '2B_diff', '3B', '3B_diff', 'HR', 'HR_diff', 'RBI', 'RBI_diff', 'SB', 'SB_diff', 'CS', 'CS_diff',
                     'BB', 'BB_diff', 'SO', 'SO_diff', 'IBB', 'IBB_diff', 'HBP', 'HBP_diff', 'SH', 'SH_diff', 'SF',
                     'SF_diff', 'GIDP', 'GIDP_diff', 'AVG', 'AVG_diff', 'OBP', 'OBP_diff', 'slugging', 'slugging_diff',
                     'OPS', 'OPS_diff', 'G_field', 'G_field_diff', 'GS_field', 'GS_field_diff', 'InnOuts_field',
                     'InnOuts_field_diff', 'PO_field', 'PO_field_diff', 'A_field', 'A_field_diff', 'E_field',
                     'E_field_diff', 'DP_field', 'DP_field_diff', 'n_awards', 'n_awards_diff', 'all_star?']

# read in data from CSVs
batting_df = pd.read_csv('data/raw/Batting.csv', usecols=column_keys + batting_stats)
fielding_df = pd.read_csv('data/raw/Fielding.csv', usecols=column_keys + fielding_stats)
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

# drop players before all-star game was inaugurated
stats_df = stats_df[stats_df['yearID'] >= first_allstar_year]
logging.info('TOTAL PLAYER SEASONS %d OR LATER: %d', first_allstar_year, len(stats_df))

# 4. number of awards (at the time of the season)
stats_df['n_awards'] = 0
for _, row in awards_df.iterrows():
    stats_df.loc[(stats_df['playerID'] == row['playerID']) & (stats_df['yearID'] > row['yearID']), ['n_awards']] += 1
logging.info('SUCCESSFULLY ADDED AWARDS STATS (ALL)')

# 5. all-star appearances
stats_df['all_star?'] = 0
for _, row in allstar_df.iterrows():
    stats_df.loc[(stats_df['playerID'] == row['playerID']) & (stats_df['yearID'] == row['yearID']), ['all_star?']] = 1
logging.info('SUCCESSFULLY ADDED ALL-STAR STATS (%s)', len(allstar_df))

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
stats_df[['AVG', 'OBP', 'slugging']] = stats_df[['AVG', 'OBP', 'slugging']].fillna(0)

# iv. on-base plus slugging
stats_df['OPS'] = stats_df['OBP'] + stats_df['slugging']

# v. differential from average of top X% (for each stat)
diff_avgs = {}
for year in range(first_allstar_year, current_year):  # set up dictionary of all top X% averages by year
    diff_avgs[year] = {}
    considered = stats_df[(stats_df['yearID'] == year) & (stats_df['G'] >= games_threshold)]
    for stat in pos_stats_to_diff:
        top_avg = considered[stat].nlargest(int(considered[stat].size * diff_percentage)).mean()
        diff_avgs[year][stat] = top_avg
    for stat in neg_stats_to_diff:
        bottom_avg = considered[stat].nsmallest(int(considered[stat].size * diff_percentage)).mean()
        diff_avgs[year][stat] = bottom_avg
    logging.debug('%d TOP %d%% AVERAGES: %s', year, int(diff_percentage * 100), diff_avgs[year])

for stat in pos_stats_to_diff + neg_stats_to_diff:
    stats_df[stat + '_diff'] = stats_df.apply(lambda row: row[stat] - diff_avgs[row['yearID']][stat], axis=1)

logging.info('SUCCESSFULLY ADDED DIFFERENTIAL STATS')

# re-order columns
stats_df = stats_df[all_stats_ordered]

# save data to CSV
stats_df.to_csv('data/preproc/stats.csv', index=False)
logging.info('SUCCESSFULLY WRITTEN DATAFRAME stats_df TO data/preproc/stats.csv')
