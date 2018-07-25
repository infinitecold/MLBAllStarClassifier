import logging
import numpy as np
import pandas as pd

# package options
logging.basicConfig(format='%(levelname)s  %(asctime)s\t%(message)s', level=logging.INFO)

pd.set_option('display.column_space', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 500)

# parameters
raw_data_directory = '../data/raw/'
output_file_path = '../data/processed/stats.csv'

column_keys = ['playerID', 'yearID']

positions_to_remove = ['P']

games_threshold = 100  # number of games a player must play in order to be considered a top player
top_players_threshold = 100  # number of top players to take average for each stat
pos_stats_to_diff = ['G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'BB', 'IBB', 'HBP', 'SH', 'SF', 'AVG', 'OBP',
                     'slugging', 'OPS', 'G_field', 'GS_field', 'InnOuts_field', 'PO_field', 'A_field', 'DP_field',
                     'n_awards']  # for these stats, the higher the stat the better
neg_stats_to_diff = ['CS', 'SO', 'GIDP', 'E_field']  # for these stats, the lower the stat the better

all_stats_ordered = ['playerID', 'yearID', 'lgID', 'POS', 'G', 'G_diff', 'AB', 'AB_diff', 'R', 'R_diff', 'H',
                     'H_diff', '2B', '2B_diff', '3B', '3B_diff', 'HR', 'HR_diff', 'RBI', 'RBI_diff', 'SB', 'SB_diff',
                     'CS', 'CS_diff', 'BB', 'BB_diff', 'SO', 'SO_diff', 'IBB', 'IBB_diff', 'HBP', 'HBP_diff', 'SH',
                     'SH_diff', 'SF', 'SF_diff', 'GIDP', 'GIDP_diff', 'AVG', 'AVG_diff', 'OBP', 'OBP_diff',
                     'slugging', 'slugging_diff', 'OPS', 'OPS_diff', 'G_field', 'G_field_diff', 'GS_field',
                     'GS_field_diff', 'InnOuts_field', 'InnOuts_field_diff', 'PO_field', 'PO_field_diff', 'A_field',
                     'A_field_diff', 'E_field', 'E_field_diff', 'DP_field', 'DP_field_diff', 'n_awards',
                     'n_awards_diff', 'all_star?']

# PROCESSING STATS FROM DATA
# 1. batting
batting_stats = ['G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF', 'GIDP']
batting_df = pd.read_csv(raw_data_directory + 'Batting.csv', usecols=column_keys + batting_stats)

batting_df = batting_df.groupby(column_keys, as_index=False).sum()

stats_df = batting_df
logging.info('SUCCESSFULLY ADDED BATTING STATS (%d)', len(batting_df))

# 2. fielding
fielding_stats = ['G', 'GS', 'InnOuts', 'PO', 'A', 'E', 'DP']
fielding_df = pd.read_csv(raw_data_directory + 'Fielding.csv', usecols=column_keys + fielding_stats)
fielding_df.columns = column_keys + [str(col) + '_field' for col in fielding_df.columns if str(col) in fielding_stats]
fielding_df = fielding_df.groupby(column_keys, as_index=False).sum()

stats_df = pd.merge(stats_df, fielding_df, how='left', on=column_keys)
logging.info('SUCCESSFULLY ADDED FIELDING STATS (%d)', len(fielding_df))

# 3. POS
appearances_df = pd.read_csv(raw_data_directory + 'Appearances.csv')
POS_df = appearances_df.groupby(column_keys, as_index=False).sum()

position_columns = ['G_p', 'G_c', 'G_1b', 'G_2b', 'G_3b', 'G_ss', 'G_lf', 'G_cf', 'G_rf', 'G_dh']
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

POS_df['POS'] = POS_df.apply(choose_position, axis=1)
POS_df = POS_df[column_keys + ['POS']]
stats_df = pd.merge(stats_df, POS_df, how='left', on=column_keys)
logging.info('SUCCESSFULLY ADDED POSITION (POS) STAT (%d)', len(POS_df))

logging.info('TOTAL PLAYER SEASONS FROM DATA: %d', len(stats_df))

# drop players with positions to be removed
stats_df = stats_df[~stats_df['POS'].isin(positions_to_remove)]
logging.info('TOTAL PLAYER SEASONS WITHOUT %s POSITIONS: %d', positions_to_remove, len(stats_df))

# drop players before all-star game was inaugurated (1933)
stats_df = stats_df[stats_df['yearID'] >= 1933]
logging.info('TOTAL PLAYER SEASONS 1933 OR LATER: %d', len(stats_df))

# 4. league (AL/NL)
league_df = appearances_df.loc[appearances_df.groupby(['playerID', 'yearID'], as_index=False, sort=False)['G_all'].idxmax()]
league_df = league_df[column_keys + ['lgID']]
stats_df = pd.merge(stats_df, league_df, how='left', on=column_keys)
logging.info('SUCCESSFULLY ADDED LEAGUE IDS (ALL)')

# 5. number of awards (at the time of the season)
awards_df = pd.read_csv(raw_data_directory + 'AwardsPlayers.csv', usecols=['playerID', 'awardID', 'yearID'])
awards_df['values'] = 1
awards_sparse = awards_df.pivot_table(index='playerID',
                                      columns='yearID',
                                      values='values',
                                      aggfunc='count',
                                      fill_value=0)  # convert to sparse matrix
matrix = np.triu(np.ones((awards_sparse.shape[1], awards_sparse.shape[1])))  # make new upper-triangular matrix of 1's
np.fill_diagonal(matrix, 0)
awards_sparse = np.matmul(awards_sparse.values, matrix)  # use matrix multiplication to accumulate values over the years
awards_df = pd.DataFrame(awards_sparse,
                         index=sorted(awards_df['playerID'].unique()),
                         columns=sorted(awards_df['yearID'].unique()))  # convert back to dense matrix
awards_df = pd.DataFrame(awards_df.stack()).reset_index()  # unstack dense matrix
awards_df.rename({'level_0': 'playerID', 'level_1': 'yearID', 0: 'n_awards'}, axis=1, inplace=True)
stats_df = pd.merge(stats_df, awards_df, how='left', on=column_keys)
stats_df['n_awards'].fillna(0, inplace=True)
logging.info('SUCCESSFULLY ADDED AWARDS STATS (ALL)')

# 6. all-star appearances
allstar_df = pd.read_csv(raw_data_directory + 'AllstarFull.csv', usecols=['playerID', 'yearID'])
allstar_df.drop_duplicates(inplace=True)
allstar_df['all_star?'] = 1
stats_df = pd.merge(stats_df, allstar_df, how='left', on=column_keys)
stats_df['all_star?'].fillna(0, inplace=True)
logging.info('SUCCESSFULLY ADDED ALL-STAR STATS (ALL: %s ALL-STARS)', len(allstar_df))

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

# v. differential from average of top players (for each stat)
diff_avgs = {}
years = stats_df['yearID'].unique().tolist()
for year in range(min(years), max(years) + 1):  # set up dictionary of averages of all top players by year
    diff_avgs[year] = {}
    considered = stats_df[(stats_df['yearID'] == year) & (stats_df['G'] >= games_threshold)]
    for stat in pos_stats_to_diff:
        top_avg = considered[stat].nlargest(top_players_threshold).mean()
        diff_avgs[year][stat] = top_avg
    for stat in neg_stats_to_diff:
        bottom_avg = considered[stat].nsmallest(top_players_threshold).mean()
        diff_avgs[year][stat] = bottom_avg
    logging.debug('%d TOP %d PLAYER AVERAGES: %s', year, top_players_threshold, diff_avgs[year])

for stat in pos_stats_to_diff + neg_stats_to_diff:
    stats_df[stat + '_diff'] = stats_df.apply(lambda row: row[stat] - diff_avgs[row['yearID']][stat], axis=1)
logging.info('SUCCESSFULLY ADDED DIFFERENTIAL STATS')

# re-order columns
stats_df = stats_df[all_stats_ordered]

# save data to CSV
stats_df.to_csv(output_file_path, index=False, float_format='%.3f')
logging.info('SUCCESSFULLY WRITTEN DATAFRAME TO %s', output_file_path)
