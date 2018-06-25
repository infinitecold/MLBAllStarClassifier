import logging
import pandas as pd
import pickle

# set options
logging.basicConfig(format='%(asctime)s %(levelname)s\t%(message)s', level=logging.INFO)

pd.set_option('display.column_space', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 500)

# read in data from CSVs
allstar_df = pd.read_csv('raw/AllstarFull.csv', usecols=['playerID', 'yearID'])
awards_df = pd.read_csv('raw/AwardsPlayers.csv', usecols=['playerID', 'awardID', 'yearID'])
batting_df = pd.read_csv('raw/Batting.csv')
fielding_df = pd.read_csv('raw/Fielding.csv', usecols=['playerID', 'yearID', 'POS', 'G', 'GS', 'InnOuts', 'PO', 'A', 'E'])
people_df = pd.read_csv('raw/People.csv', usecols=['playerID', 'nameFirst', 'nameLast'])
logging.info('SUCCESSFULLY LOADED CSVS INTO DATAFRAMES')

# parameters
player_stats = {}
batting_stats = ['G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF']
fielding_stats = ['G', 'GS', 'InnOuts', 'PO', 'A', 'E']
positions_to_remove = ['P', 'C']
eras = {'-1919': (0, 1919), '1920-1941': (1920, 1941), '1942-1945': (1942, 1945), '1946-1962': (1946, 1962),
        '1963-1976': (1963, 1976), '1977-1992': (1977, 1992), '1993-2009': (1993, 2009), '2010-': (2010, 3000)}

# PROCESSING STATS FROM DATA
# 1. batting
for _, row in batting_df.iterrows():
    player_season = (row['playerID'], row['yearID'])
    if player_season not in player_stats:
        player_stats[player_season] = {}
        for stat in batting_stats:
            player_stats[player_season][stat] = row[stat]
    else:
        for stat in batting_stats:
            player_stats[player_season][stat] += row[stat]
logging.info('SUCCESSFULLY ADDED BATTING STATS')

# 2. fielding
player_positions = {}
for _, row in fielding_df.iterrows():
    player_season = (row['playerID'], row['yearID'])
    if player_season not in player_stats:
        player_stats[player_season] = {}
    if player_season not in player_positions:
        player_positions[player_season] = {}
        for stat in fielding_stats:
            player_stats[player_season][stat + '_field'] = row[stat]
    else:
        for stat in fielding_stats:
            player_stats[player_season][stat + '_field'] += row[stat]
    player_positions[player_season][row['POS']] = row['G']
logging.info('SUCCESSFULLY ADDED FIELDING STATS')

logging.info('NUMBER OF PLAYER SEASONS: %d', len(player_stats))
for player_season in player_positions:  # remove players who played >90% games as pitchers or catchers
    for position in positions_to_remove:
        if position in player_positions[player_season] and player_positions[player_season][position] / sum(player_positions[player_season].values()) > 0.9:
            player_stats.pop(player_season, None)
            logging.debug('PLAYER %s (%s) REMOVED WITH >90%% %s POSITION', player_season[0], player_season[1], position)
logging.info('NUMBER OF PLAYER SEASONS WITHOUT %s POSITIONS: %d', positions_to_remove, len(player_stats))

# build stats_df
stats_df = pd.DataFrame.from_dict(player_stats, orient='index')
stats_df.reset_index(inplace=True)
stats_df.rename(columns={'level_0': 'playerID', 'level_1': 'yearID'}, inplace=True)

# 3. number of awards (at the time of the season)
stats_df['n_awards'] = 0
for _, row in awards_df.iterrows():
    stats_df.loc[(stats_df['playerID'] == row['playerID']) & (stats_df['yearID'] > row['yearID']), ['n_awards']] += 1
logging.info('SUCCESSFULLY ADDED AWARDS STATS')

# 4. eras (-1919, 1920-1941, 1942-1945, 1946-1962, 1963-1976, 1977-1992, 1993-2009, 2010-)
for era_label in eras:
    stats_df[era_label] = 0
for label, era in eras.iteritems():
    stats_df.loc[(stats_df['yearID'] >= era[0]) & (stats_df['yearID'] <= era[1]), [label]] = 1
logging.info('SUCCESSFULLY ADDED ERA STATS')

# 5. all-star appearances
stats_df['all_star?'] = 0
for _, row in allstar_df.iterrows():
    stats_df.loc[(stats_df['playerID'] == row['playerID']) & (stats_df['yearID'] == row['yearID']), ['all_star?']] = 1
logging.info('SUCCESSFULLY ADDED ALL-STAR STATS')

# ADDING STATS USING EXISTING DATA
stats_df['AVG'] = stats_df['H'] / stats_df['AB']  # batting average

PA_no_SH = stats_df['AB'] + stats_df['BB'] + stats_df['HBP'] + stats_df['SF']  # plate appearances without SH
stats_df['OBP'] = (stats_df['H'] + stats_df['BB'] + stats_df['HBP']) / PA_no_SH  # on-base percentage

singles = stats_df['H'] - stats_df['2B'] - stats_df['3B'] - stats_df['HR']
stats_df['slugging'] = (stats_df['HR'] * 4 + stats_df['3B'] * 3 + stats_df['2B'] * 2 + singles) / stats_df['AB']

stats_df['OPS'] = stats_df['OBP'] + stats_df['slugging']

# fill missing data (NaN values) with 0s
stats_df.fillna(0, inplace=True)

# re-order columns
stats_df = stats_df[['playerID', 'yearID', 'G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'BB', 'SO', 'IBB', 'HBP',
                     'SH', 'SF', 'AVG', 'OBP', 'slugging', 'OPS', 'G_field', 'GS_field', 'InnOuts_field', 'PO_field',
                     'A_field', 'E_field', 'n_awards', '-1919', '1920-1941', '1942-1945', '1946-1962', '1963-1976',
                     '1977-1992', '1993-2009', '2010-', 'all_star?']]

# log first 50 rows
logging.debug(stats_df.head(50))

# SAVING DATA TO CSVS AND PICKLES
stats_df.to_csv('preprocessed/stats.csv', index=False)
logging.info('SUCCESSFULLY WRITTEN DATAFRAME stats_df TO CSV preprocessed/stats.csv')
