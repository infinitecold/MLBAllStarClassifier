import logging
import pandas as pd

# set options
logging.basicConfig(format='%(asctime)s %(levelname)s\t%(message)s', level=logging.INFO)

pd.set_option('display.column_space', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 500)

# read in data from CSVs
allstar_df = pd.read_csv('data/AllstarFull.csv', usecols=['playerID', 'yearID'])
awards_df = pd.read_csv('data/AwardsPlayers.csv', usecols=['playerID', 'awardID', 'yearID'])
batting_df = pd.read_csv('data/Batting.csv')
fielding_df = pd.read_csv('data/Fielding.csv', usecols=['playerID', 'yearID', 'POS', 'G', 'GS', 'InnOuts', 'PO', 'A', 'E', 'DP'])
people_df = pd.read_csv('data/People.csv', usecols=['playerID', 'nameFirst', 'nameLast'])
logging.info('SUCCESSFULLY LOADED CSVS INTO DATAFRAMES')

# congregate data into dictionary
player_stats = {}

# 1. batting
for _, row in batting_df.iterrows():
    player_season = (row['playerID'], row['yearID'])
    if player_season not in player_stats:
        player_stats[player_season] = {}
    for stat in ['G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF']:
        player_stats[player_season][stat] = row[stat]

# 2. fielding
player_positions = {}
for _, row in fielding_df.iterrows():
    player_season = (row['playerID'], row['yearID'])
    if player_season not in player_stats:
        player_stats[player_season] = {}
    if player_season not in player_positions:
        player_positions[player_season] = {}
        for stat in ['G', 'GS', 'InnOuts', 'PO', 'A', 'E', 'DP']:
            player_stats[player_season][stat + '_field'] = row[stat]
    else:
        for stat in ['G', 'GS', 'InnOuts', 'PO', 'A', 'E', 'DP']:
            player_stats[player_season][stat + '_field'] += row[stat]
    player_positions[player_season][row['POS']] = row['G']
logging.info('NUMBER OF PLAYER SEASONS: %d', len(player_stats))

for player_season in player_positions:  # remove players who played >90% games as pitchers or catchers
    for position in ['P', 'C']:
        if position in player_positions[player_season] and player_positions[player_season][position] / sum(player_positions[player_season].values()) > 0.9:
            player_stats.pop(player_season, None)
            logging.info('PLAYER %s (%s) REMOVED WITH >90%% %s POSITION', player_season[0], player_season[1], position)
logging.info('NUMBER OF PLAYER SEASONS WITHOUT P\'S AND C\'S: %d', len(player_stats))

# build stats_df
stats_df = pd.DataFrame.from_dict(player_stats, orient='index')
stats_df.reset_index(inplace=True)
stats_df.rename(columns={'level_0': 'playerID', 'level_1': 'yearID'}, inplace=True)

# 3. number of awards (at the time of the season)
stats_df['n_awards'] = 0
for _, row in awards_df.iterrows():
    stats_df.loc[(stats_df['playerID'] == row['playerID']) & (stats_df['yearID'] > row['yearID']), ['n_awards']] += 1

# 4. eras (-1919, 1920-1941, 1942-1945, 1946-1962, 1963-1976, 1977-1992, 1993-2009, 2010-)
for era in ['-1919', '1920-1941', '1942-1945', '1946-1962', '1963-1976', '1977-1992', '1993-2009', '2010-']:
    stats_df[era] = 0

stats_df.loc[stats_df['yearID'] < 1920, ['-1919']] = 1
stats_df.loc[(stats_df['yearID'] >= 1920) & (stats_df['yearID'] <= 1941), ['1920-1941']] = 1
stats_df.loc[(stats_df['yearID'] >= 1942) & (stats_df['yearID'] <= 1945), ['1942-1945']] = 1
stats_df.loc[(stats_df['yearID'] >= 1946) & (stats_df['yearID'] <= 1962), ['1946-1962']] = 1
stats_df.loc[(stats_df['yearID'] >= 1963) & (stats_df['yearID'] <= 1976), ['1963-1976']] = 1
stats_df.loc[(stats_df['yearID'] >= 1977) & (stats_df['yearID'] <= 1992), ['1977-1992']] = 1
stats_df.loc[(stats_df['yearID'] >= 1993) & (stats_df['yearID'] <= 2009), ['1993-2009']] = 1
stats_df.loc[stats_df['yearID'] > 2009, ['2010-']] = 1

# 5. all-star appearances
stats_df['all_star?'] = 0
for _, row in allstar_df.iterrows():
    stats_df.loc[(stats_df['playerID'] == row['playerID']) & (stats_df['yearID'] == row['yearID']), ['all_star?']] = 1

# fill missing data (NaN values) with 0s
stats_df.fillna(0, inplace=True)

logging.info('stats_df:\n%s', stats_df.head(40))
